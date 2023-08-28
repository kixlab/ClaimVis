import sys
sys.path.append('../Gloc')
import spacy
from spacy import displacy
from datetime import datetime
from nltk.corpus import wordnet as wn
from nl4dv import NL4DV
from nl4dv.utils import helpers
from nl4dv.utils.constants import attribute_types
from Gloc.utils.llm import *
from rapidfuzz import fuzz
import pandas as pd
import json
import nltk
import re
from models import *
from Gloc.processor.ans_parser import AnsParser
from DataMatching import DataMatcher
from collections import defaultdict

class AutomatedViz(object):
    def __init__(
            self, 
            matcher: DataMatcher = None,
            datasrc: str = None, 
            table: dict or pd.DataFrame = None, 
            attributes: str = None, 
            test: bool = False,
            value_map: dict = None
        ):
        self.datasrc = datasrc

        # lower case the whole dataset if it's a test
        if isinstance(table, dict):
            self.table = table["data"]  
            self.table_name = table["name"] 
        elif isinstance(table, pd.DataFrame):
            self.table = table
            self.table_name = "table"
        else: # load from csv
            self.table = pd.read_csv(self.datasrc)
            self.table_name = "table"

        if test:
            self.table.columns = self.table.columns.str.lower()
            self.table = self.table.applymap(lambda s:s.lower() if type(s) == str else s)
        
        self.attributes = attributes or list(self.table.columns)
        self.value_map = value_map or dict()

        # initialize AnsParser
        self.parser = AnsParser()
        self.datamatcher = matcher or DataMatcher()

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
            {"role": "system", "content": """Given a list of attributes and a claim, please wrap the relevant references in the claim to the attributes with curly braces and return a map of references to the MOST SIMILAR attributes. For example, if the claim is 'The United State has the highest energy consumption in 2022.', and the attributes are ['country', 'energy consumption per capita', 'year'], then the output should be 
             {
                "wrap": 'The {United State} has the highest {energy consumption} in {2022}.',
                "map": {
                    "United State": "country",
                    "energy consumption": "energy consumption per capita",
                    "2022": "year"
                }
            }
            DO NOT CHANGE or ADD any word within the wrap text except for curly braces.
            DO NOT CREATE new attributes that are not in the list of attributes."""},
            {"role": "user", "content": f"claim: {text}\nattributes: {self.attributes}"},
        ]
        response = call_model(
                        model=Model.GPT4,
                        prompt=message,
                        temperature=0,
                        max_decode_steps=200,
                        samples=1
                    )[0]
        # parse the response (not sure if this works for all cases)
        response_dict = json.loads(response)
        for ref, attr in response_dict['map'].copy().items():
            flag = False
            for value in self.table[attr]:
                if fuzz.ratio(ref, str(value)) > 0.8:
                    flag = True
                    break
            if not flag and fuzz.ratio(ref, attr) <= 0.8 \
                and self.datamatcher.similarity_score(ref, attr) <= 0.5:
                response_dict['wrap'] = response_dict['wrap'].replace(f'{{{ref}}}', f'{ref}')
                response_dict['map'].pop(ref)
            
        return response_dict
    
    def retrieve_data_points(self, text: str, verbose: bool = False):
        tag_map = self.tag_attribute_gpt(text)
        if verbose: print(f"tagmap: {tag_map}")

        def isAny(attr, func: callable):
            return any(func(val) for val in self.table[attr].to_list())

        # infer nominal, temporal, and quantitative attributes
        dates, fields, categories = None, set(), []
        for ref, attr in tag_map['map'].items():
            if helpers.isdate(ref)[0] and self.datamatcher.attr_score_batch(attr, ['time', 'year', 'date']) > 0.5:
                dates = {
                    "value": attr,
                    "range": self.table[attr].to_list()
                }
                fields.add(Field(
                                name=attr,
                                type="temporal",
                                timeUnit= self.parser.parse_unit(ref) or "year"
                            ))  
            elif helpers.isint(ref) or helpers.isfloat(ref) or isAny(attr, helpers.isint) or isAny(attr, helpers.isfloat):
                categories.append({
                    'table_name': self.table_name,
                    'label': ref,
                    'value': attr,
                    'unit': self.parser.parse_unit(attr) or self.table[attr].dtype.name,
                    'provenance': ""
                })
            else: # nominal
                fields.add(Field( name=attr, type="nominal" ))      
            
        
        if verbose: print(f"fields: {fields}, map: {self.value_map.keys()}")
        assert len(fields) == len(self.value_map.keys()), "The number of fields should be the same as the number of values in the value map."
        filtered_table = self.table
        for field in fields:
            if field.type == "nominal":
                filtered_table = filtered_table[filtered_table[field.name].isin(self.value_map[field.name])]
            elif field.type == "temporal":
                if len(self.value_map[field.name]) == 1:
                    value = self.value_map[field.name].pop()
                    # Convert the value to the same type as the values in the table
                    value = type(self.table[field.name].iloc[0])(value)
                    filtered_table = filtered_table[filtered_table[field.name] == value]
                else: # more than one
                    assert len(self.value_map[field.name]) >= 2, "The number of values in the value map should be more than 1."
                    # Convert the values to the same type as the values in the table
                    values = [type(self.table[field.name].iloc[0])(v) for v in self.value_map[field.name]]
                    old, recent = min(values), max(values)
                    filtered_table = filtered_table[(filtered_table[field.name] >= old) & (filtered_table[field.name] <= recent)]

        if verbose:
            # print(f"dates: {dates}")
            print(f"filtered table: {filtered_table}")
            # print(f"fields: {fields}")
            print(f"categories: {categories}")

        # final pass to retrieve all datapoints
        datapoints, data_fields = [], list(map(lambda x: x.name, fields))
        for category in categories:
            for _, row in filtered_table[data_fields + [category['value']]].iterrows():
                dataPoint = DataPointValue(
                    tableName=self.table_name,
                    valueName=category['value'],
                    fields={attr: row[attr] for attr in data_fields},
                    unit=category['unit'],
                    value=row[category['value']]
                )

                if dates:
                    dataPoint.fields['date'] = str(row[dates['value']])
                datapoints.append(dataPoint)
        
        # replace all the wrap text with attribute names
        for ref, attr in tag_map['map'].items():
            tag_map['wrap'] = tag_map['wrap'].replace(f'{{{ref}}}', f'{{{attr}}}')
        
        newSet = DataPointSet(
                statement=tag_map['wrap'],
                dataPoints=datapoints,
                fields=fields,
                ranges=Ranges(
                    values = categories,
                    fields = {attr: list(set(self.table[attr].to_list())) for attr in data_fields}
                ),
                tableName=self.table_name
            )
        # fetch the name of the temporal field
        temporal_field_name = [field.name for field in fields if field.type == "temporal"][0]
        newSet.ranges.fields[temporal_field_name] = DateRange(
            date_start={
                'label': str(min(dates['range'])), 
                'value': str(min(dates['range']))
            },
            date_end={
                'label': str(max(dates['range'])),
                'value': str(max(dates['range']))
            }
        ) if dates else None
        
        return [newSet]

if __name__ == "__main__":
    # tag_date_time("Some people are crazy enough to get out in the winter, especially november and december where it's freezing code outside.")
    data = pd.read_csv("../Datasets/owid-energy-data.csv").iloc[:5]
    vizPipeline = AutomatedViz(
                    # datasrc="../Datasets/owid-energy-data.csv",
                    table=data,
                    attributes=['primary_energy_consumption', 'year', 'country', 'coal_share_energy']
                )

    t = vizPipeline.retrieve_data_points("In America, unemployment rate of males are higher than that of females.")
    print(t)
    
