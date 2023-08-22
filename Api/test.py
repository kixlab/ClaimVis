import spacy
from spacy import displacy
from datetime import datetime
from nltk.corpus import wordnet as wn
from pycorenlp import StanfordCoreNLP
from nl4dv import NL4DV
from nl4dv.utils import helpers
from nl4dv.utils.constants import attribute_types
import json
import nltk

class AutomatedViz(object):
    def __init__(self, datasrc: str = None):
        self.datasrc = datasrc
        label_attribute = None
        dependency_parser_config = {
                "name": "corenlp-server", 
                "url": "http://localhost:9000",
            }

        self.nl4dv = NL4DV(
                        verbose=False, 
                        debug=True, 
                        data_url=self.datasrc, 
                        label_attribute=label_attribute, 
                        dependency_parser_config=dependency_parser_config
                    )
        self.query_processor = self.nl4dv.query_genie_instance
        self.attribute_processor = self.nl4dv.attribute_genie_instance
        self.data_processor = self.nl4dv.data_genie_instance

    def tag_date_time(self, text: str):
        # Parse date time from the claim
        # Use NLP libraries to extract date from user_claim
        nlp = spacy.load("en_core_web_sm")
        
        doc = nlp(text)

        replaced_text = ""
        for token in doc:
            if token.ent_type_ == "DATE":
                replaced_text += "{date} "
            else:
                replaced_text += token.text + " "

        print(replaced_text.strip())

    def get_query_ngram_spans(self, query: str, n: int = 5):
        def get_ngrams(input, n):
            input = input.split(' ')
            output = []
            for i in range(len(input)-n+1):
                output.append((input[i:i+n], i, i+n-1))
            return output

        query_alpha_str = ''.join([i for i in query if not i.isdigit()])
        ngrams = dict()
        for i in range(len(query_alpha_str.split()), 0, -1):
            for ngram, start, end in get_ngrams(query_alpha_str, i):
                ngram_str = ((' '.join(map(str, ngram))).rstrip()).lower()
                ngrams[ngram_str] = dict()
                ngrams[ngram_str]['raw'] = ngram
                ngrams[ngram_str]['lower'] = ngram_str
                ngrams[ngram_str]['stemmed_lower'] = ' '.join(self.nl4dv.porter_stemmer_instance.stem(t) for t in nltk.word_tokenize(ngram_str))
                ngrams[ngram_str]['span'] = (start, end)

        return ngrams

    def tag_attribute(self, text: str):
        toks = text.split(' ')
        attributes = self.data_processor.data_attribute_map

        ngrams = self.get_query_ngram_spans(text)
        extracted_attributes = self.attribute_processor.extract_attributes(ngrams)
        print(extracted_attributes)
        
        idxlist = []
        for attr, attr_details in extracted_attributes.items():
            for phrase in attr_details['queryPhrase']:
                sp, ep = ngrams[phrase]['span']
                idxlist.append((sp, ep, attr))
        
        idxlist.sort(key=lambda x: x[0], reverse=True)
        for sp, ep, attr in idxlist:
            if attributes[attr]['dataType'] in [attribute_types['NOMINAL'], attribute_types['TEMPORAL']]:
                toks[sp:ep+1] = [f'{{{attr}}}']
            else:
                toks[sp:ep+1] = ['{value}']

        return ' '.join(toks)

if __name__ == "__main__":
    # tag_date_time("Some people are crazy enough to get out in the winter, especially november and december where it's freezing code outside.")
    vizPipeline = AutomatedViz(datasrc="../Datasets/movies-w-year.csv")
    # ngrams = vizPipeline.get_query_ngram("show the rating of movies with budget larger than one hundred millions.")
    # print(ngrams.keys())
    t = vizPipeline.tag_attribute("show the content rating of The Phantom in 2019.")
    print(t)
    print(vizPipeline.tag_date_time(t))
