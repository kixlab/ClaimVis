import sys
sys.path.append("../Gloc")
sys.path.append("..")

from Gloc.utils.normalizer import post_process_sql
from Pipeline.DataMatching import DataMatcher
from Gloc.nsql.database import NeuralDB
from Gloc.processor.ans_parser import AnsParser
from AutomatedViz import AutomatedViz
from llama_index.query_engine import PandasQueryEngine
import openai
import pandas as pd
import os

class Tester():
    def __init__(self, datasrc: str = None):
        self.datasrc = datasrc

    def test_post_process_sql(self):
        sql = """No, we cannot determine if the share of children in work dropped by one percentage point between 2012 and 2016 as the 
data provided does not include the necessary information """
        table = pd.read_csv(os.path.join(self.datasrc, "Social Development.csv"))
        # table.columns = table.columns.str.lower()
        # table = table.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        table.reset_index(inplace=True)
        table.rename(columns={'index': 'row_id'}, inplace=True)
        matcher = DataMatcher()

        new_sql, value_map = post_process_sql(
            sql_str=sql,
            df=table,
            verbose=True,
            matcher=matcher
        )
        print(new_sql, value_map)
    
    def test_filter_data(self):
        table = pd.read_csv(os.path.join(self.datasrc, "Social Protection & Labor.csv"))
        # table.columns = table.columns.str.lower()
        # table = table.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        table = table[table['country_name'].isin(['united states', 'china'])]
        table = table[table['date'].isin([2022])]
        print(table)
    
    def test_retrieve_data_points(self):
        table = pd.read_csv(os.path.join(self.datasrc, "Private Sector.csv"))
        db = NeuralDB(
            tables=[table],
            add_row_id=True,
            normalize=False,
            lower_case=True
        )
        sql = """SELECT case when ( "Children in employment, total (% of children ages 7-14)" - lag ( "Children in employment, total (% of children ages 7-14)" ) over ( ORDER by "date" ) ) = -1 then \'Yes\' else \'No\' end FROM w WHERE "date" IN ( 2012 , 2016 ) """
        print(db.execute_query(sql))

    def test_parse_ans(self):
        parser = AnsParser()
        # message = """SELECT "country" , "Annual total production-based emissions of carbon dioxide (CO₂), including land-use change (million tonnes)" FROM w WHERE "country" IS NOT null and "Annual total production-based emissions of carbon dioxide (CO₂), including land-use change (million tonnes)" IS NOT null and "Annual total production-based emissions of carbon dioxide (CO₂), including land-use change (million tonnes)" <> \'nan\'"""
        message = """country" , "Annual total production-based emissions of carbon dioxide (CO₂), including land-use change (million tonnes)"""
        # print(parser.parse_sql_unit(message))
        print(parser.parse_unit(message))

    def test_tag_date_time(self):
        import spacy
        query = "China had the best coal production against VAL_1 in 2011."
        if query.endswith(('.', '?')): # remove the period at the end
            query = query[:-1]
        # Load the spacy model
        nlp = spacy.load("en_core_web_sm")
        # Create a Doc object
        doc = nlp(query)
        # Dependency parsing
        for token in doc:
            print(token.text, token.pos_, token.ent_type_, token.dep_, token.head.text, token.head.pos_,
                [child for child in token.children])
            
        # start_default, end_default = 2010, 2020
        # dates = []
        # for ind, token in enumerate(doc):
        #     if token.ent_type_ == 'DATE' and token.head.pos_ in ['ADP', 'SCONJ']:
        #         dates.append(ind)
        # print(dates)
        
        # if len(dates) == 0: # add the most recent date
        #     query += f" in {end_default}"
        # elif len(dates) == 1: # rules
        #     ind = dates[0]-1
        #     if doc[ind].text.lower() in ['since', 'from']:
        #         query = f"{doc[:ind]} from {doc[ind+1]} to {end_default} {doc[ind+2:]}"
        #     elif doc[ind].text.lower() in ['til', 'until']:
        #         query = f"{doc[:ind]} from {start_default} {doc[ind:]}"
        
        print(query)
    
    def test_tag_attr_gpt(self):
        datasrc = "../Datasets/owid-co2.csv"
        autoviz = AutomatedViz(attributes=['year', 'Population of each country or region', 'country'], datasrc=datasrc)
        tag_map = autoviz.tag_attribute_gpt("Show the population of China and the USA in 2020.")
        print(tag_map)

    def test_llama_index(self):
        table = pd.read_csv(os.path.join(self.datasrc, "Private Sector.csv"))
        engine = PandasQueryEngine(table)
        response = engine.query("China has the highest Import value index (2000= 100).")
        print(response)

    def test_open_finetune_gpt(self):
        file_path = "../Gloc/generation/finetune/claim_tagging.jsonl"
        t = openai.File.create(
            file = open(file_path, 'rb'),
            purpose = 'fine-tune',
            user_provided_filename= "claim_tagging_fine_tune_2"
        )
        print(t)

    def test_list_files(self):
        files = openai.File.list()
        print(files)
    
    def test_create_fine_tune(self):
        # t = openai.FineTuningJob.create(
        #     training_file="file-QB3dm9RtoIwykKuLJrjLwVP2",
        #     model = 'gpt-3.5-turbo'
        # )
        # print(t)
        t = openai.FineTuningJob.list()
        print(t)
    
    def test_infer_country(self):
        from TableReasoning import TableReasoner
        dm = DataMatcher(datasrc="../Datasets")
        tb = TableReasoner(datamatcher=dm)
        pass
    

if __name__ == "__main__":
    tester = Tester(datasrc="../Datasets")
    tester.test_tag_date_time()
    # pass