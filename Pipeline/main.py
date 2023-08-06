"""
    This file integrates all parts of the pipeline.
"""

from TableReasoning import TableReasoning
from Summarizer import Summarizer
from ClaimDetection import ClaimDetector
from DataMatching import DataMatcher

class Pipeline(object):
    def __init__(self, datasrc: str = None):
        self.table_reasoner = TableReasoning()
        self.summarizer = Summarizer()
        self.claim_detector = ClaimDetector()
        self.data_matcher = DataMatcher(datasrc=datasrc)
    
    def _parse_sentences(self, text: str):
        # parse sentences from text
        sentences = text.split(".")
        return sentences

    def run(self, text: str):
        # parse sentences from text
        sentences = self._parse_sentences(text)

        # detect claims
        for sentence in sentences:
            score = self.claim_detector.detect(sentence)
            if score > .5:
                # find top k datasets
                top_k_datasets = self.data_matcher.find_top_k_datasets(sentence, k=2)

                # summarize the datasets
                for dataset in top_k_datasets:
                    summary = self.summarizer.summarize(dataset[1])
                    # reason the summary
                    self.table_reasoner.reason(summary)