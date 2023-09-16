import copy
import os
import pandas as pd
from typing import Dict, List
import uuid
import duckdb
from Pipeline_async.models import Dataset
from Pipeline_async.DataMatching import DataMatcher

from utils.normalizer import convert_df_type, process_raw_table
# from utils.mmqa.image_stuff import get_caption

def check_in_and_return(key: str, source: dict):
    # `` wrapped means as a whole
    if key.startswith("`") and key.endswith("`"):
        key = key[1:-1]
    if key in source.keys():
        return source[key]
    else:
        for _k, _v in source.items():
            if _k.lower() == key.lower():
                return _v
        raise ValueError("{} not in {}".format(key, source))

class NeuralDB(object):
    def __init__(
            self, 
            tables: list[dict or pd.DataFrame],
            add_row_id=True, 
            normalize=False, 
            lower_case=True
        ):
        global w # this is the name of universal table we will query

        assert len(tables) == 1, "NeuralDB only supports one table for now."
        self.tables = [process_raw_table(
                            tables[0],
                            add_row_id=add_row_id,
                            normalize=normalize,
                            lower_case=lower_case
                        )]
        # a trick to allow duckdb to read the table from class variable
        w = self.tables[0]['table']
        self.table_name = "w"

    def __str__(self):
        return str(self.execute_query("SELECT * FROM {}".format(self.table_name)))

    def get_table(self, table_name=None):
        table_name = self.table_name if not table_name else table_name
        sql_query = "SELECT * FROM {}".format(table_name)
        _table = self.execute_query(sql_query)
        return _table

    def get_header(self, table_name=None):
        _table = self.get_table(table_name)
        return _table['header']

    def get_rows(self, table_name):
        _table = self.get_table(table_name)
        return _table['rows']

    def get_table_df(self):
        return self.tables[0]['table']

    def get_table_title(self):
        return self.tables[0]['title']
    
    def save_table(self, path):
        self.get_table_df().to_csv(path, index=False)

    def update_table(self, attributes: set):
        """
        Filter the table with the relevant attributes.
        @param attributes: set
        @return: pd.DataFrame
        """
        global w
        try:
            self.tables[0]['table'] = w = self.get_table_df()[list(attributes)]
        except KeyError:
            raise KeyError("Attributes {} not in table.".format(attributes))

    def execute_query(self, sql_query: str, return_dict=False):
        """
        Basic operation. Execute the sql query on the database we hold.
        @param sql_query:
        @return:
        """
        # When the sql query is a column name (@deprecated: or a certain value with '' and "" surrounded).
        if len(sql_query.split(' ')) == 1 or (sql_query.startswith('`') and sql_query.endswith('`')):
            col_name = sql_query # single word, which is a column name
            new_sql_query = r"SELECT row_id, {} FROM {}".format(col_name, self.table_name)
            # Here we use a hack that when a value is surrounded by '' or "", the sql will return a column of the value,
            # while for variable, no ''/"" surrounded, this sql will query for the column.
            out = duckdb.sql(new_sql_query)
        # When the sql query wants all cols or col_id, which is no need for us to add 'row_id'.
        elif sql_query.lower().startswith("select *") or sql_query.startswith("select col_id"):
            out = duckdb.sql(sql_query)
        else:
            try:
                # SELECT row_id in addition, needed for result and old table alignment.
                new_sql_query = "SELECT row_id, " + sql_query[7:]
                out = duckdb.sql(new_sql_query)
            except Exception as e:
                # Execute normal SQL, and in this case the row_id is actually in no need.
                out = duckdb.sql(sql_query)

        result = out.df()

        if return_dict:
            return {
                "header": result.columns.tolist(), 
                "rows": result.values.tolist()
            }
        else:
            return result

    def add_sub_table(self, sub_table, table_name=None, verbose=True):
        """
        Add sub_table into the table.
        @return:
        """
        table_name = self.table_name if not table_name else table_name
        sql_query = "SELECT * FROM {}".format(table_name)
        oring_table = self.execute_query(sql_query)
        old_table = pd.DataFrame(oring_table["rows"], columns=oring_table["header"])
        # concat the new column into old table
        sub_table_df_normed = convert_df_type(pd.DataFrame(data=sub_table['rows'], columns=sub_table['header']))
        new_table = old_table.merge(sub_table_df_normed,
                                    how='left', on='row_id')  # do left join
        new_table.to_sql(table_name, self.sqlite_conn, if_exists='replace',
                         index=False)
        if verbose:
            print("Insert column(s) {} (dtypes: {}) into table.\n".format(', '.join([_ for _ in sub_table['header']]),
                                                                          sub_table_df_normed.dtypes))
            

class MultiNeuralDB(object):
    def __init__(
            self, datasets: list[Dataset],
            dm: DataMatcher
        ):
        self.tables = {}
        for dataset in datasets:
            data_file, country_attr, date_attr = dm.load_table(dataset.name, dataset.fields, infer_date_and_country=True)
            self.tables[dataset.name] = {
                "data": data_file,
                "country": country_attr,
                "date": date_attr,
                "fields": dataset.fields,
            }

    

        
