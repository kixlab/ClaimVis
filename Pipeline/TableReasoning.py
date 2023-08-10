# CoT few-shot prompting 
import sys
sys.path.append("../Gloc")
sys.path.append("..")

import pandas as pd
from generation.claimvis_prompt import Prompter, TemplateKey
from utils.llm import *
from processor.ans_parser import AnsParser
from common.functionlog import log_decorator
from utils.normalizer import post_process_sql
import re

class TableReasoner(object):
    def __init__(self):
        self.prompter = Prompter()
        self.parser = AnsParser()

    def _call_api_1(
        self: object, 
        question: str,
        template_key: TemplateKey,
        table: pd.DataFrame = None
    ):
        """
            Call API for few-shot prompting
            Input: question, template_key, table
            Output: response
        """
        prompt = self.prompter.build_prompt(
            template_key=template_key,
            table=table,
            question=question
        )

        response = call_model(
            model=Model.GPT3,
            use_code=False,
            max_decode_steps=500,
            temperature=0,
            prompt=prompt,
            samples=1
        )

        return response
        
    def _call_api_2(self, prompt: list):
        """
            Call API
            Input: prompt
            Output: response
        """
        response = call_model(
            model=Model.GPT3,
            use_code=False,
            temperature=0,
            max_decode_steps=500,
            prompt=prompt,
            samples=1
        )

        return response

    def _suggest_queries(self, claim: str):
        """
            Suggest queries given a claim
            Input: claim
            Output: list of suggested queries
        """
        # suggest different queries in form of "[{query: ...}, {query: ...}}]"
        suggestions = self._call_api_1(
            question=claim,
            template_key=TemplateKey.QUERY_GENERATION_2
        )[0]

        return self.parser.parse_gen_query(suggestions)
    
    def _decompose_query(self, query: str):
        """
            Decompose query into subqueries
            Input: query
            Output: list of subqueries
        """
        decomposed_ans = self._call_api_1(
            question=query,
            template_key=TemplateKey.QUERY_DECOMPOSE
        )[0]

        return self.parser.parse_dec_reasoning(decomposed_ans)
    
    def _decompose_table(self, claim: str, table: pd.DataFrame):
        """
            Decompose table into subtable
            Input: claim, table
            Output: subtable
        """
        decomposed_cols = self._call_api_1(
            question=claim,
            template_key=TemplateKey.COL_DECOMPOSE,
            table=table
        )[0]

        cols = self.parser.parse_col_dec(decomposed_cols)
        return table.loc[:, cols]

    def _generate_sql(self, claim: str, table: pd.DataFrame):
        """
            Generate SQL query
            Input: claim, table
            Output: SQL query
        """
        sql = self._call_api_1(
            question=claim,
            template_key=TemplateKey.SQL_GENERATION,
            table=table
        )[0]

        return self.parser.parse_sql(sql)

    def _generate_nsql(self, claim: str, table: pd.DataFrame):
        """
            Generate NSQL query
            Input: claim, table
            Output: NSQL query
        """
        nsql = self._call_api_1(
            question=claim,
            template_key=TemplateKey.NSQL_GENERATION,
            table=table
        )[0]

        refined_nsql = self.parser.parse_nsql(nsql)
        return post_process_sql(
            sql_str=refined_nsql,
            df=table
        )

    @log_decorator
    def reason_1st_query(self, claim: str, table: pd.DataFrame):
        """
            Reasoning pipeline for CoT
            Input: claim, table
            Output: justification
        """
        
        # take first query from suggested queries
        suggestions = self._suggest_queries(claim)
        print(f"generated queries: {suggestions}")
        first_query = suggestions[0]

        # decompose queries
        sub_queries = self._decompose_query(first_query)
        print(f"steps of reasoning: {sub_queries}")

        # build prompt for decomposed subqueries uptil the last query
        dec_prompt = self.prompter.build_prompt(
            template_key=TemplateKey.DEC_REASONING_2,
            table=table,
            question=first_query,
        )
        content = "\n".join(f"Q{str(i+1)}: {query}" for i, query in enumerate(sub_queries))
        dec_prompt.append({
            "role": "user", 
            "content": content
        })

        # # call API for decomposed reasoning
        # response = self._call_api_2(dec_prompt)

        # # final query
        # dec_prompt.extend([{
        #     "role": "assistant", 
        #     "content": response
        #     },{
        #     "role": "user",
        #     "content": f"Q{len(sub_queries)}: {sub_queries[-1]}"
        # }])

        print(f"full prompt:\n{dec_prompt}")

        answer = self._call_api_2(dec_prompt)

        return answer

if __name__ == "__main__":
    table_reasoner = TableReasoner()
    claim = "The top 2 countries total gdp do not surpass 300000."
    df = pd.read_csv("../Datasets/owid-energy-data.csv")
    print(table_reasoner.reason_1st_query(claim, df))
    # print(table_reasoner._call_api_1(
    #         question=claim,
    #         template_key=TemplateKey.QUERY_GENERATION
    #     )[0])
