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

class Pipeline(object):
    def __init__(self, datasrc: str = None):
        self.table_reasoner = TableReasoner()
        # self.summarizer = Summarizer()
        self.claim_detector = ClaimDetector()
        self.data_matcher = DataMatcher(datasrc=datasrc)

        self.datasrc = datasrc
        with open("viz_trials/index.txt", "r") as f:
            self.trials = int(f.read().strip())
        atexit.register(self.clean_up)
    
    def clean_up(self):
        with open("viz_trials/index.txt", "w") as f:
            f.write(str(self.trials))
    
    def run(self, text: str, THRE_SHOLD: float = .5, verbose: bool = True):
        # parse sentences from text
        sentences = sent_tokenize(text)

        # detect claims
        claim_map, claims = defaultdict(list), []
        for sentence in sentences:
            claim, score = self.claim_detector.detect(sentence, verbose=verbose)
            if score > THRE_SHOLD:
                if verbose: print(f"claim: {claim}")
                # find top k datasets
                top_k_datasets = self.data_matcher.find_top_k_datasets(claim, k=1)
                if verbose: print(f"top k datasets: {top_k_datasets}")

                # reason the claim
                for dataset, des, similarity in top_k_datasets:
                    claim_map[claim].append(
                        self.table_reasoner.reason(
                            claim=sentence,
                            table=pd.read_csv(f"{self.datasrc}/{dataset}"),
                            verbose=verbose,
                            fuzzy_match=True
                        ))
                    
            claims.append(claim)
                    
        return claim_map, claims
    
    def create_trial(self, claim:str):
        pipeline = Pipeline(datasrc="../Datasets")
        claim_map, claims = pipeline.run(claim)
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

if __name__ == "__main__":
    pipeline = Pipeline(datasrc="../Datasets")
    text = "The economy of China is larger than that of France.."
    
    # try:
    pipeline.run(text)
    # except Exception as e:
    #     print(e)
        