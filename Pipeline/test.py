import sys
sys.path.append("../Gloc")
sys.path.append("..")

from Gloc.utils.normalizer import post_process_sql
import pandas as pd
from Pipeline.DataMatching import DataMatcher
from Gloc.nsql.database import NeuralDB
from Gloc.processor.ans_parser import AnsParser
import os

class Tester():
    def __init__(self, datasrc: str = None):
        self.datasrc = datasrc

    def test_post_process_sql(self):
        sql = """SELECT case when ( "Fertility rate, total (births per woman)" < ( "Wanted fertility rate (births per woman)" ) / 3 ) then \'Yes\' else \'No\' end AS is_less FROM w WHERE "country_name" = \'Republic of Korea\' and "date" = extract ( year FROM current_date ) """
        table = pd.read_csv(os.path.join(self.datasrc, "Gender.csv"))
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
        sql = """SELECT "date" , "Merchandise exports (current US$)" FROM w WHERE ( "country_name" = \'China\' or "country_name" = \'United States\' ) and "date" BETWEEN 2011 and 2022 GROUP by "date" having MAX ( "Merchandise exports (current US$)" ) = "Merchandise exports (current US$)" """
        print(db.execute_query(sql))

    def test_parse_ans(self):
        parser = AnsParser()
        message = """SELECT "People using at least basic drinking water services", "hugo(% of population)" FROM w WHERE "date" = 2020"""
        print(parser.parse_sql_unit(message))

    def test_tag_date_time(self):
        import spacy
        query = "The current fertility rate of Korea is near one third of the replacement rate."
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
            
        start_default, end_default = 2010, 2020
        dates = []
        for ind, token in enumerate(doc):
            if token.ent_type_ == 'DATE' and token.head.pos_ in ['ADP', 'SCONJ']:
                dates.append(ind)
        print(dates)
        
        if len(dates) == 0: # add the most recent date
            query += f" in {end_default}"
        elif len(dates) == 1: # rules
            ind = dates[0]-1
            if doc[ind].text.lower() in ['since', 'from']:
                query = f"{doc[:ind]} from {doc[ind+1]} to {end_default} {doc[ind+2:]}"
            elif doc[ind].text.lower() in ['til', 'until']:
                query = f"{doc[:ind]} from {start_default} {doc[ind:]}"
        
        print(query)

if __name__ == "__main__":
    tester = Tester(datasrc="../Datasets")
    tester.test_tag_date_time()
    # pass