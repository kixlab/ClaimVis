"""
    This file integrates all parts of the pipeline.
"""

from TableReasoning import TableReasoner
from Summarizer import Summarizer
from ClaimDetection import ClaimDetector
from DataMatching import DataMatcher
import pandas as pd
from nltk.tokenize import sent_tokenize
from collections import defaultdict

from lida.modules import Manager
from lida.datamodel import Goal
import lida
import pandas   as pd

import base64
from io import BytesIO
from PIL import Image
import json
import os
import atexit
from rapidfuzz import fuzz
from models import UserClaimBody
from Gloc.generation.claimvis_prompt import TemplateKey
import re

class Pipeline(object):
    def __init__(self, datasrc: str = None):
        # self.summarizer = Summarizer()
        self.claim_detector = ClaimDetector()
        self.data_matcher = DataMatcher(datasrc=datasrc)
        self.table_reasoner = TableReasoner(datamatcher=self.data_matcher)

        self.datasrc = datasrc
        with open("viz_trials/index.txt", "r") as f:
            self.trials = int(f.read().strip())
        atexit.register(self.clean_up)
    
    def clean_up(self):
        with open("viz_trials/index.txt", "w") as f:
            f.write(str(self.trials))

    def detect_claim(
            self, claim:str, 
            llm_classify:bool = False, 
            verbose: bool = True, 
            boundary_extract:bool=False,
            score_threshold: float = 0.5
        ):
        return self.claim_detector.detect(
                    claim, verbose=verbose, 
                    llm_classify=llm_classify,
                    boundary_extract=boundary_extract,
                    score_threshold=score_threshold
                )
    
    def find_top_k_datasets(
            self, 
            claim:str, 
            k: int = 1, 
            method: str = "attr", 
            verbose: bool = True
        ):
        return self.data_matcher.find_top_k_datasets(claim, k=k, method=method, verbose=verbose)
    
    def extract_claims(self, body: UserClaimBody or str):
        if isinstance(body, str):
            return sent_tokenize(body)
        
        userClaim, paragraph = body.userClaim, body.paragraph
        if not paragraph:
            return [userClaim]
        
        prompter = self.table_reasoner.prompter
        # extract claims and disambiguate from paragraph
        prompt = prompter.build_prompt(
                            template_key=TemplateKey.CLAIM_EXTRACTION,
                            table=None,
                            paragraph=paragraph,
                            userClaim=userClaim
                        )
        result = self.table_reasoner._call_api_2(prompt=prompt)[0]

        match = re.search(r'"Claims": (\[.*?\])', result, re.DOTALL)
        return json.loads(match.group(1)) if match else []
    
    def reason(
            self, claim: str,
            dataset: str, 
            relevant_attrs:list=[], 
            fuzzy_match: str=True, 
            verbose: bool = True
        ):
        table = pd.read_csv(f"{self.datasrc}/{dataset}")
        table.name = dataset
        reason_map = self.table_reasoner.reason(
                        claim=claim,
                        table=table,
                        verbose=verbose,
                        fuzzy_match=fuzzy_match,
                        more_attrs=relevant_attrs,
                    )
        reason_map["sub_table"]["name"] = dataset
        return reason_map
    
    def run_on_text(
            self, text: str or UserClaimBody, 
            THRE_SHOLD: float = .5, 
            verbose: bool = True
        ):
        """
        This function runs the pipeline on the given text (multiple sentences) or UserClaimBody (paragraph and sentence).

        Parameters:
            text (str): The text to run the pipeline on.
            THRE_SHOLD (float): The threshold for claim detection. Defaults to 0.5.
            verbose (bool): Whether to print verbose output. Defaults to True.

        Returns:
            tuple: A tuple containing the claim map and the list of claims.
        """

        claim_map, claims = defaultdict(list), []
        for sentence in self.extract_claims(text):
            claim, score = self.detect_claim(sentence, verbose=verbose, llm_classify=True, score_threshold=THRE_SHOLD)
            if score > THRE_SHOLD:
                if verbose: print(f"claim: {claim}")
                # find top k datasets
                top_k_datasets = self.find_top_k_datasets(claim, verbose=verbose)

                # reason the claim
                for dataset, des, similarity, relevant_attrs in top_k_datasets:
                    claim_map[claim].append(
                        self.reason(
                            claim=claim,
                            dataset=dataset,
                            relevant_attrs=relevant_attrs,
                            fuzzy_match=True,
                            verbose=verbose
                        )
                    )
                    
            claims.append(claim)
                    
        return claim_map, claims
    
    def create_trial(self, claim:str):
        """
        This function creates a visualization trial for a given claim. It uses the pipeline to reason the claim and generate a visualization based on LIDA.

        Parameters:
            claim (str): The claim to create a trial for.

        Returns:
            None
        """
        
        pipeline = Pipeline(datasrc="../Datasets")
        claim_map, claims = pipeline.run_on_text(claim)
        reason = claim_map[claims[0]][0] # only take the first dataset
        vis_task, sub_table = reason["suggestions"][0]["visualization"], reason["sub_table"]

        trial_path = f"viz_trials/trial{self.trials}"
        if os.path.exists(trial_path):
            raise ValueError("Folder already exists.")
        else:
            os.makedirs(trial_path)
        
        # save subtable to csv
        data_url=f"{trial_path}/sub_table.csv"
        sub_table.to_csv(data_url)

        # perform trial
        os.environ["LIDA_ALLOW_CODE_EVAL"] = "1"
        lida = Manager()

        summary = lida.summarize(data_url)
        goal = Goal(
                    question=vis_task + ". Sample only 50 datapoints when there are more than 50.",
                    index=0,
                    visualization="",
                    rationale=""
                )
        
        # generate code specifications for charts
        vis_specs = lida.generate_viz(summary=summary, goal=goal, library="seaborn") # altair, matplotlib etc
        # execute code to return charts (raster images or other formats)
        charts = lida.execute_viz(code_specs=vis_specs, data=pd.read_csv(data_url), summary=summary)
        print(charts[0].code, "\n**********")

        def decode_base64_to_image(base64_string):
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            return image

        decoded_image = decode_base64_to_image(charts[0].raster)
        decoded_image.show()
        # Save the image to trial{idx}.png
        decoded_image.save(f"{trial_path}/trial{self.trials}.png")

        # save only the orginal claim and vistask to vega.json
        vega = {
            "claim": claim,
            "vis_task": vis_task
        }
        with open(f"{trial_path}/vega.json", 'w') as outfile:
            json.dump(vega, outfile)

        self.trials += 1

def main():
    pipeline = Pipeline(datasrc="../Datasets")
    text = "The country's imports and exports rank 1st in the world, accounting for more than 12% of total global trade."
    paragraph = "China has been the world's largest exporter of goods since 2009. Official estimates suggest Chinese exports amounted to $2.097 trillion in 2017. Since 2013, China has also become the largest trading nation in the world. The country's imports and exports rank 1st in the world, accounting for more than 12% of total global trade. China is also the world's second-largest importer and the second-largest foreign investor. China is a member of numerous formal and informal multilateral organizations, including the WTO, APEC, BRICS, the Shanghai Cooperation Organization (SCO), the BCIM and the G20. Using a PPP exchange rate of 1 yuan = US$0.15 (2017 Annual Average) China's total GDP in 2017 was US$23.12 trillion. In 2018, China's autonomous regions had the highest nominal GDP per capita, with Shanghai at US$25,383, followed by Beijing at US$22,914, Tianjin at US$21,724, and Jiangsu at US$20,753."
    
    x = pipeline.extract_claims(UserClaimBody(userClaim=text, paragraph=paragraph))
    print(x)

def profile_func(func):
    import cProfile
    import pstats

    # Create a profiler
    profiler = cProfile.Profile()

    # Run the function you want to profile
    profiler.runcall(func)

    # Create a Stats object to format and print the profiler's data
    stats = pstats.Stats(profiler)

    # Sort the data by the cumulative time spent in the function
    stats.sort_stats('cumulative')

    # Print the profiling data
    stats.print_stats(50)


if __name__ == "__main__":
    from TableReasoning import main as table_reasoner_main
    from DataMatching import main as data_matcher_main
    from api import main as api_main

    profile_func(api_main) 
    # main()
    # table_reasoner_main()
    # print(fuzz.ratio("pg13", "pg-13"))