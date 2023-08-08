import re
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
        return re.search(r'SQL: "(.*?)"', message).group(1)
    
    def parse_nsql(self, message: str):
        return re.search(r'NeuralSQL: "(.*?)"', message).group(1)