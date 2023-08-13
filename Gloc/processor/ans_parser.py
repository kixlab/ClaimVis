import re
import pandas as pd
from Gloc.nsql.parser import extract_answers

class AnsParser(object):
    def __init__(self) -> None:
        pass

    def parse_dec_reasoning(self, message: str):
        matches = re.findall("\".*?\"", message)
        if not matches: # empty list --> 
            return matches
        # move the first match to the tail due to it being the last query in order
        return [match[1:-1] for match in matches[1:]] + [matches[0][1:-1]]
    
    def parse_row_dec(self, message: str):
        match = re.search(r'f_row\(\[(.*?)\]\)', message)
        if match:
            return [row.strip() for row in match.group(1).split(',')]
        else:
            return []        

    def parse_col_dec(self, message: str):
        match = re.search(r'f_col\(\[(.*?)\]\)', message)
        if match:
            return [col.strip() for col in match.group(1).split(',')]
        else:
            return []        
    
    def parse_gen_query(self, message: str):
        return re.findall(r'query: "(.*?)"', message)
    
    def parse_sql(self, message: str):
        match = re.search(r'SQL: (.*)', message)
        
        return match.group(1) if match else None
    
    def parse_sql_2(self, message: str):
        strs = message.split('\n')
        return [re.search(r'A\d+: (.*)', s).group(1) for s in strs]
    
    def parse_nsql(self, message: str):
        match = re.search(r'NeuralSQL: (.*)', message)
        
        return match.group(1) if match else None
    
    def parse_sql_result(self, sub_table: pd.DataFrame or dict):
        if isinstance(sub_table, dict):
            return extract_answers(sub_table)
        else: # is dataframe
            if sub_table.empty or sub_table.columns.empty:
                return []
            answer = []
            if 'row_id' in sub_table.columns:
                for _, row in sub_table.iterrows():
                    answer.extend(row.values[1:])
                return answer
            else:
                for _, row in sub_table.iterrows():
                    answer.extend(row.values)
                return answer
    