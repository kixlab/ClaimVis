import sys
sys.path.append("../Gloc")
sys.path.append("..")

from Gloc.utils.normalizer import post_process_sql
import pandas as pd
import os

class Tester():
    def __init__(self, datasrc: str = None):
        self.datasrc = datasrc

    def test_post_process_sql_2(self):
        sql = "SELECT coal_production FROM w WHERE country = 'Indiae' and year = 2011"
        table = pd.read_csv(os.path.join(self.datasrc, "owid-energy-data.csv"))

        new_sql = post_process_sql(
            sql_str=sql,
            df=table,
            verbose=True
        )
        print(new_sql)

if __name__ == "__main__":
    # tester = Tester(datasrc="../Datasets")
    # tester.test_post_process_sql_2()
    pass