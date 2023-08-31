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
        sql = """SELECT SUM ( "People using at least basic drinking water services (% of population)" ) FROM w WHERE "date" = 2020 """
        table = pd.read_csv(os.path.join(self.datasrc, "Environment.csv"))
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


if __name__ == "__main__":
    tester = Tester(datasrc="../Datasets")
    tester.test_parse_ans()
    # pass