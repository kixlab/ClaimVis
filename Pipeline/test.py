import sys
sys.path.append("../Gloc")
sys.path.append("..")

from Gloc.utils.normalizer import post_process_sql
import pandas as pd
from Pipeline.DataMatching import DataMatcher
from Gloc.nsql.database import NeuralDB
import os

class Tester():
    def __init__(self, datasrc: str = None):
        self.datasrc = datasrc

    def test_post_process_sql(self):
        sql = """SELECT "coal_production" FROM w WHERE "country" = \'US\' AND "year" = 2022"""
        table = pd.read_csv(os.path.join(self.datasrc, "owid-energy-data.csv"))
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
        table.columns = table.columns.str.lower()
        table = table.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        table = table[table['country_name'].isin(['united states', 'china'])]
        table = table[table['date'].isin([2022])]
        print(table)
    
    def test_retrieve_data_points(self):
        db = NeuralDB(
            tables=[pd.read_csv(os.path.join(self.datasrc, "Social Protection & Labor.csv"))],
            add_row_id=True,
            normalize=False,
            lower_case=True
        )
        sql = """SELECT "unemployment, female (% of female labor force) (national estimate)" FROM w WHERE "country_name" = \'United States\' and "date" = 2022"""
        print(db.execute_query(sql))


if __name__ == "__main__":
    tester = Tester(datasrc="../Datasets")
    tester.test_post_process_sql()
    # pass