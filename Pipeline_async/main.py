"""
    This file integrates all parts of the pipeline.
"""

import asyncio
import time

from TableReasoning import TableReasoner
from ClaimDetection import ClaimDetector
from DataMatching import DataMatcher
import pandas as pd
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import pandas   as pd
import json
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

    async def detect_claim(
            self, claim:str, 
            llm_classify:bool = False, 
            verbose: bool = True, 
            boundary_extract:bool=False,
            score_threshold: float = 0.5
        ):
        return await self.claim_detector.detect(
                    claim, verbose=verbose, 
                    llm_classify=llm_classify,
                    boundary_extract=boundary_extract,
                    score_threshold=score_threshold
                )
    
    async def find_top_k_datasets(
            self, 
            claim:str, 
            k: int = 1, 
            method: str = "attr", 
            verbose: bool = True
        ):
        return await self.data_matcher.find_top_k_datasets(claim, k=k, method=method, verbose=verbose)
    
    async def extract_claims(self, body: UserClaimBody or str):
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
        response = await self.table_reasoner._call_api_2(prompt=prompt, model="gpt-3.5-turbo")
        result = response[0]

        match = re.search(r'"Claims": (\[.*?\])', result, re.DOTALL)
        return json.loads(match.group(1)) if match else []
    
    async def reason(
            self, claim: str,
            dataset: str, 
            relevant_attrs:list=[], 
            fuzzy_match: str=True, 
            verbose: bool = True
        ):
        table = pd.read_csv(f"{self.datasrc}/{dataset}")
        table.name = dataset
        reason_map = await self.table_reasoner.reason(
                        claim=claim,
                        table=table,
                        verbose=verbose,
                        fuzzy_match=fuzzy_match,
                        more_attrs=relevant_attrs,
                    )
        reason_map["sub_table"]["name"] = dataset
        return reason_map
    
    async def run_on_text(
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
        for sentence in await self.extract_claims(text):
            claim, score = await self.detect_claim(sentence, verbose=verbose, llm_classify=False, score_threshold=THRE_SHOLD)
            if score > THRE_SHOLD:
                if verbose: print(f"claim: {claim}")
                # find top k datasets
                top_k_datasets = await self.find_top_k_datasets(claim, verbose=verbose)

                # reason the claim
                for dataset, des, similarity, relevant_attrs in top_k_datasets:
                    claim_map[claim].append(
                        await self.reason(
                                claim=claim,
                                dataset=dataset,
                                relevant_attrs=relevant_attrs,
                                fuzzy_match=True,
                                verbose=verbose
                            )
                    )
                    
            claims.append(claim)
                    
        return claim_map, claims

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

    # profile_func(api_main) 
    start = time.perf_counter()
    asyncio.run(table_reasoner_main())
    end = time.perf_counter()
    print(f"Table Reasoner: {end - start:0.4f} seconds")