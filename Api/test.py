import sys
sys.path.append('../Gloc')
import spacy
from spacy import displacy
from datetime import datetime
from nltk.corpus import wordnet as wn
from pycorenlp import StanfordCoreNLP
from nl4dv import NL4DV
from nl4dv.utils import helpers
from nl4dv.utils.constants import attribute_types
from Gloc.utils.llm import *
from fuzzywuzzy import fuzz
import pandas as pd
import json
import nltk
import re
from models import *
from Gloc.processor.ans_parser import AnsParser

class AutomatedViz(object):
    def __init__(self, datasrc: str = None, table: pd.DataFrame = None, attributes: str = None):
        self.datasrc = datasrc

        # lower case the whole dataset
        self.table = table if table is not None else pd.read_csv(datasrc)
        self.table.columns = self.table.columns.str.lower()
        self.table = self.table.applymap(lambda s:s.lower() if type(s) == str else s)
        
        self.attributes = attributes or list(self.table.columns)
        
        # initialize NL4DV
        label_attribute = None
        dependency_parser_config = {
                "name": "corenlp-server", 
                "url": "http://localhost:9000",
            }

        self.nl4dv = NL4DV(
                        verbose=False, 
                        debug=True, 
                        data_url=self.datasrc, 
                        data_value=table,
                        label_attribute=label_attribute, 
                        dependency_parser_config=dependency_parser_config
                    )
        self.query_processor = self.nl4dv.query_genie_instance
        self.attribute_processor = self.nl4dv.attribute_genie_instance
        self.data_processor = self.nl4dv.data_genie_instance

        # initialize AnsParser
        self.parser = AnsParser()

    def tag_date_time(self, text: str, verbose: bool = False):
        # Parse date time from the claim
        nlp = spacy.load("en_core_web_sm")
        
        doc = nlp(text)

        replaced_text = ""
        for token in doc:
            if token.ent_type_ == "DATE":
                replaced_text += "{date} "
            else:
                replaced_text += token.text + " "

        if verbose: print(replaced_text.strip())
        return replaced_text

    def get_query_ngram_spans(self, query: str, n: int = 5):
        def get_ngrams(input, n):
            input = input.split(' ')
            output = []
            for i in range(len(input)-n+1):
                output.append((input[i:i+n], i, i+n-1))
            return output

        query_alpha_str = ''.join([i for i in query if not i.isdigit()])
        ngrams = dict()
        for i in range(n or len(query_alpha_str.split()), 0, -1):
            for ngram, start, end in get_ngrams(query_alpha_str, i):
                ngram_str = ((' '.join(map(str, ngram))).rstrip()).lower()
                ngrams[ngram_str] = dict()
                ngrams[ngram_str]['raw'] = ngram
                ngrams[ngram_str]['lower'] = ngram_str
                ngrams[ngram_str]['stemmed_lower'] = ' '.join(self.nl4dv.porter_stemmer_instance.stem(t) for t in nltk.word_tokenize(ngram_str))
                ngrams[ngram_str]['span'] = (start, end)

        return ngrams

    def tag_attribute_nl4dv(self, text: str, verbose: bool = False):
        toks = text.split(' ')
        attributes = self.data_processor.data_attribute_map

        ngrams = self.get_query_ngram_spans(text)
        extracted_attributes = self.attribute_processor.extract_attributes(ngrams)
        if verbose: print(extracted_attributes)
        
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

    def tag_attribute_gpt(self, text: str):
        message = [
            {"role": "system", "content": """Given a list of attributes and a claim, please wrap the relevant references in the claim to the attributes with curly braces and return a map of references to corresponding attributes. For example, if the claim is 'The United State has the highest energy consumption in 2022.', and the attributes are ['country', 'energy consumption per capita', 'year'], then the output should be 
             {
                "wrap": 'The {United State} has the highest {energy consumption} in {2022}.',
                "map": {
                    "United State": "country",
                    "energy consumption": "energy consumption per capita",
                    "2022": "year"
                }
            }
            DO NOT CHANGE or ADD any word within the wrap except for curly braces."""},
            {"role": "user", "content": f"claim: {text.lower()}\nattributes: {self.attributes}"},
        ]
        response = call_model(
                        model=Model.GPT4,
                        prompt=message,
                        temperature=0,
                        max_decode_steps=200,
                        samples=1
                    )[0]
        # print(response)

        # parse the response
        response_dict = json.loads(response)
        for ref, attr in response_dict['map'].copy().items():
            flag = False
            for value in self.table[attr]:
                if fuzz.ratio(ref, str(value)) > 0.8:
                    flag = True
                    break
            if not flag and fuzz.ratio(ref, attr) <= 0.8:
                response_dict['wrap'] = response_dict['wrap'].replace(f'{{{ref}}}', f'{ref}')
                response_dict['map'].pop(ref)
            
        return response_dict
    
    def retrieve_data_points(self, text: str, verbose: bool = False):
        # tag_map = self.tag_attribute_gpt(text)
        tag_map = {
                    "wrap": 'The {United State} has the highest {energy consumption} in {2022}.',
                    "map": {
                        "United State": "country",
                        "energy consumption": "primary_energy_consumption",
                        "2022": "year"
                    }
                }

        def isAny(attr, func: callable):
            if verbose:
                print(f"attribute: {attr}. func: {func}")
                print(any(func(val) for val in self.table[attr].to_list()))
            return any(func(val) for val in self.table[attr].to_list())

        dates, fields, categories, datapoints = None, [], [], []
        for ref, attr in tag_map['map'].items():
            if helpers.isdate(ref)[0]:
                dates = {
                    "name": attr,
                    "range": self.table[attr].to_list()
                }
                fields.append(Field(
                                name=attr,
                                type="temporal",
                                timeUnit= self.parser.parse_unit(ref) or "year"
                            ))  
            elif helpers.isint(ref) or helpers.isfloat(ref) or isAny(attr, helpers.isint) or isAny(attr, helpers.isfloat):
                categories.append({
                    'label': ref,
                    'value': ref,
                    'unit': self.parser.parse_unit(attr) or self.table[attr].dtype.name,
                    'provenance': None,
                    'table_name': None
                })
            else: # nominal
                fields.append(Field(
                                name=attr,
                                type="nominal"
                            ))      
            
        return dates, fields, categories
        # final pass to retrieve all datapoints
        
        # for idx, value in enumerate(self.table[attr].to_list()):
        #     datapoints.append(DataPointValue(
        #         tableName=None,
        #         date=dates['range'][idx] if dates else None,
        #         category=ref,
        #         otherFields={attr: value},
        #         unit=self.parser.parse_unit(attr) or self.table[attr].dtype.name,
        #         value=value
        #     ))

if __name__ == "__main__":
    # tag_date_time("Some people are crazy enough to get out in the winter, especially november and december where it's freezing code outside.")
    vizPipeline = AutomatedViz(
                    datasrc="../Datasets/owid-energy-data.csv",
                    attributes=['primary_energy_consumption', 'year', 'country']
                )

    # vizPipeline.retrieve_data_points("The United State has the highest energy consumption in the world in 2020.")
    x, y, z = vizPipeline.retrieve_data_points("any thing")
    print(y, z)
    