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