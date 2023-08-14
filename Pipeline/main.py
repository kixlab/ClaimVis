"""
    This file integrates all parts of the pipeline.
"""

from TableReasoning import TableReasoner
from Summarizer import Summarizer
from ClaimDetection import ClaimDetector
from DataMatching import DataMatcher
import pandas as pd
from nltk.tokenize import sent_tokenize
from nl4dv import NL4DV
import json

class Pipeline(object):
    def __init__(self, datasrc: str = None):
        self.table_reasoner = TableReasoner()
        # self.summarizer = Summarizer()
        self.claim_detector = ClaimDetector()
        self.data_matcher = DataMatcher(datasrc=datasrc)

        self.datasrc = datasrc
    
    def run(self, text: str, THRE_SHOLD: float = .5, verbose: bool = True):
        # parse sentences from text
        sentences = sent_tokenize(text)

        # detect claims
        claim_map, claims = {}, []
        for sentence in sentences:
            claim, score = self.claim_detector.detect(sentence, verbose=verbose)
            claims.append(claim)
            claim_map[claim] = []
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
                            verbose=False,
                            fuzzy_match=False
                        ))
                    
        return claim_map, claims
    

if __name__ == "__main__":
    pipeline = Pipeline(datasrc="../Datasets")
    text = "The Phantom is the movie with highest budget of all time."
    claim_map, claims = pipeline.run(text)
    for claim in claims:
        # save subtable to csv
        data_url="temp/sub_table.csv"
        claim_map[claim][0]["sub_table"].to_csv(data_url)        
        label_attribute = None
        dependency_parser_config = {
                        "name": "corenlp-server", 
                        "url": "http://localhost:9000",
                    }

        nl4dv_instance = NL4DV(verbose=False, 
                            debug=True, 
                            data_url=data_url, 
                            label_attribute=label_attribute, 
                            dependency_parser_config=dependency_parser_config
                            )
        vega = nl4dv_instance.analyze_query(claim_map[claim][0]["suggestions"][0]["visualization"])
        with open('temp/vega.json', 'w') as json_file:
            json.dump(vega, json_file)
        

        # print(nl4dv_response)