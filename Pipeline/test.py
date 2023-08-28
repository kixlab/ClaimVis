import sys
sys.path.append("../Gloc")
sys.path.append("..")

from Gloc.utils.normalizer import post_process_sql
import pandas as pd
from Pipeline.DataMatching import DataMatcher
import os

class Tester():
    def __init__(self, datasrc: str = None):
        self.datasrc = datasrc

    def test_post_process_sql(self):
        sql = """SELECT "electric power consumption (kwh per capita)" FROM w WHERE "country_name"= \'America\' and "date" = 2011"""
        table = pd.read_csv(os.path.join(self.datasrc, "Energy & Mining.csv"))
        table.columns = table.columns.str.lower()
        table = table.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        table.reset_index(inplace=True)
        table.rename(columns={'index': 'row_id'}, inplace=True)
        matcher = DataMatcher()

        new_sql = post_process_sql(
            sql_str=sql,
            df=table,
            verbose=True,
            matcher=matcher
        )
        print(new_sql)
    
    def test_filter_data(self):
        table = pd.read_csv(os.path.join(self.datasrc, "Social Protection & Labor.csv"))
        table.columns = table.columns.str.lower()
        table = table.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        table = table[table['country_name'].isin(['united states', 'china'])]
        table = table[table['date'].isin([2022])]
        print(table)


if __name__ == "__main__":
    tester = Tester(datasrc="../Datasets")
    tester.test_filter_data()
    # pass