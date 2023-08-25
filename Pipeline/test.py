import sys
sys.path.append("../Gloc")
sys.path.append("..")

from Gloc.utils.normalizer import post_process_sql
import pandas as pd
import os

class Tester():
    def __init__(self, datasrc: str = None):
        self.datasrc = datasrc

    def test_post_process_sql(self):
        sql = """SELECT "birth rate, crude (per 1,000 people)" FROM w WHERE "country_name" = 'Mexico' AND "date" = 2022"""
        table = pd.read_csv(os.path.join(self.datasrc, "Health.csv"))
        table.columns = table.columns.str.lower()
        table = table.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        table.reset_index(inplace=True)
        table.rename(columns={'index': 'row_id'}, inplace=True)

        new_sql = post_process_sql(
            sql_str=sql,
            df=table,
            verbose=True
        )
        print(new_sql)

if __name__ == "__main__":
    tester = Tester(datasrc="../Datasets")
    tester.test_post_process_sql()
    # pass