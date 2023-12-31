import re
import pandas as pd
from Gloc.nsql.parser import extract_answers
from functools import reduce

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
        queries = re.findall(r'query: "(.*?)"', message)
        vis_tasks = re.findall(r'vis: "(.*?)"', message)
        reasons = re.findall(r'explain: "(.*?)"', message)

        if "attributes" in message:
            # prompt with context
            attr_strs = re.findall(r'attributes: \[(.*?)\]', message)
            attributes = set(reduce(lambda x, y: x + y, [attr_str.split(', ') for attr_str in attr_strs], []))
            attributes = [attr.strip("\"") for attr in attributes]
        else:
            # prompt without context
            attributes = None

        return queries, vis_tasks, reasons, attributes
    
    def parse_sql(self, message: str):
        match = re.search(r'SQL: (.*)', message)
        
        return match.group(1) if match else None
    
    def parse_sql_2(self, message: str):
        ans = []
        for s in message.split('\n'):
            match = re.search(r'A\d+: (.*)', s)
            if match: ans.append(match.group(1))
        return ans
    
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
    
    def parse_evaluation(self, message: str):
        match = re.search(r'revised: "(.*?)"', message)
        # return the whole message if no match
        if match: return match.group(1)
        match = re.search(r'"revised": "(.*?)"', message)
        return match.group(1) if match else message
    
    def parse_unit(self, message: str):
        matches = re.findall(r'\((.*?)\)', message)
        return matches[-1] if matches else None
    
    def parse_sql_unit(self, sql: str):
        try:
            # sql has the form of "SELECT ... FROM table WHERE ..."
            match = re.match(r'SELECT (.*?) FROM', sql).group(1)
            # The unit can be in the form <AGGREGATION> ( <COLUMN> ) or <COLUMN>
            # Paying attention to spaces
            match = re.search(r'(\w+)\s*\(\s*(.*)\s*\)|"(.*)"', match)
            if match.group(1):
                agg, col_ = match.group(1), match.group(2)
                if col_ == "*":
                    col = "all"
                else: 
                    col = col_[1:-1]
            else:
                agg, col = "None", match.group(3)
            
            unit = self.parse_unit(col) or col # if no unit, return the whole column
            return f"Unit: {unit}. Aggregation: {agg}"
        except:
            return "Unit: None. Aggregation: None"

    def parse_gpt_list(self, message: str):
        # gpt list has form of: 1. <item1> 2. <item2> ...
        matches = re.findall(r'\d+\.\s*(.*?)\s*\n', message)
        return matches