# CoT few-shot prompting 
import sys
sys.path.append("../Gloc")
sys.path.append("..")

import pandas as pd
from Gloc.generation.claimvis_prompt import Prompter, TemplateKey

# this import also acquires log_decor, don't import it again
from Gloc.utils.llm import *

from Gloc.processor.ans_parser import AnsParser
from Gloc.utils.normalizer import post_process_sql
from Gloc.nsql.database import NeuralDB
from Gloc.utils.utils import majority_vote

class TableReasoner(object):
    def __init__(
            self, 
            temperature=0.0, 
            max_decode_steps=500, 
            samples=1, 
            model=Model.GPT3
        ):
        self.prompter = Prompter()
        self.parser = AnsParser()

        self.temperature = temperature # to change
        self.max_decode_steps = max_decode_steps # fixed
        self.samples = samples # to change
        self.model = model # fixed
        
    def _call_api_1(
        self: object, 
        question: str,
        template_key: TemplateKey,
        table: pd.DataFrame = None,
        samples: int = -1,
        temperature: float = -1,
    ):
        """
            Call API for few-shot prompting using a question, a template, and a table 
            Input: question, template_key, table
            Output: prompt, response
        """
        prompt = self.prompter.build_prompt(
            template_key=template_key,
            table=table,
            question=question
        )

        response = call_model(
            model=Model.GPT3,
            max_decode_steps=self.max_decode_steps,
            temperature=temperature if temperature > 0 else self.temperature,
            prompt=prompt,
            samples=samples if samples > 0 else self.samples
        )

        return prompt, response
        
    def _call_api_2(
            self, 
            prompt: list,
            temperature: float = -1,
            samples: int = -1
        ):
        """
            Call API using a provide prompt
            Input: prompt
            Output: response
        """
        response = call_model(
            model=Model.GPT3,
            temperature=temperature if temperature > 0 else self.temperature,
            max_decode_steps=self.max_decode_steps,
            prompt=prompt,
            samples=samples if samples > 0 else self.samples
        )

        return response

    @log_decorator
    def _suggest_queries(self, claim: str, table: pd.DataFrame=None):
        """
            Suggest queries given a claim (and a table)
            Input: claim
            Output: list of suggested queries
        """
        # suggest different queries in form of "[{query: ...}, {query: ...}}]"
        template = TemplateKey.QUERY_GENERATION_2 if table is None else TemplateKey.QUERY_GENERATION_3
        _, suggestions = self._call_api_1(
            question=claim,
            template_key=template,
            table=table
        )

        return self.parser.parse_gen_query(suggestions[0])
    
    def _decompose_query(self, query: str):
        """
            Decompose query into subqueries
            Input: query
            Output: list of subqueries
        """
        _, decomposed_ans = self._call_api_1(
            question=query,
            template_key=TemplateKey.QUERY_DECOMPOSE
        )

        return self.parser.parse_dec_reasoning(decomposed_ans[0])
    
    def _decompose_colunns(self, claim: str, table: pd.DataFrame):
        """
            Decompose table into subtable using column decomposition
            Input: claim, table
            Output: subtable
        """
        _, decomposed_cols = self._call_api_1(
            question=claim,
            template_key=TemplateKey.COL_DECOMPOSE,
            table=table
        )

        cols = self.parser.parse_col_dec(decomposed_cols[0])
        return table.loc[:, cols]

    def _generate_sql(
            self, 
            query: str, 
            table: pd.DataFrame,
            template_key: TemplateKey,
            samples: int = 5,
            temperature: float = 0.4,
            fuzzy_match: bool = False
        ):
        """
            Generate SQL queries based on the provided query and table.
            The type of SQL generation is determined by the template_key.
            The number of samples and the temperature can be adjusted.
            If fuzzy_match is set to True, the function will return post-processed SQL queries.

            Parameters:
                query (str): The query based on which SQL queries are generated.
                table (pd.DataFrame): The table used for SQL generation.
                template_key (TemplateKey): The key determining the type of SQL generation.
                samples (int, optional): The number of samples to generate. Defaults to 5.
                temperature (float, optional): The temperature for generation. Defaults to 0.4.
                fuzzy_match (bool, optional): Whether to return post-processed SQL queries. Defaults to False.

            Returns:
                list: A list of generated SQL queries.
        """
        
        if template_key not in [TemplateKey.NSQL_GENERATION, TemplateKey.SQL_GENERATION, TemplateKey.SQL_GENERATION_2]:
            raise ValueError("Invalid template key for SQL generation")
        
        _, sqls = self._call_api_1(
                        question=query,
                        template_key=template_key,
                        table=table,
                        samples=samples if samples > 0 else self.samples, # need samples to aggregate
                        temperature=temperature if temperature > 0 else self.temperature # need some creativity
                    )

        if template_key == TemplateKey.NSQL_GENERATION:
            psqls = [self.parser.parse_nsql(sql) for sql in sqls]
        elif template_key == TemplateKey.SQL_GENERATION:
            psqls = [self.parser.parse_sql(sql) for sql in sqls]
        elif template_key == TemplateKey.SQL_GENERATION_2:
            psqls = [self.parser.parse_sql_2(sql) for sql in sqls]

        if fuzzy_match:
            return [post_process_sql(
                        sql_str=psql, 
                        table=table,
                        fuzzy_match=True,
                        verbose=True
                    ) for psql in psqls]
        else:
            return psqls
    
    def _exec_sqls_from_sub_queries(
            self,
            db: NeuralDB,
            table: pd.DataFrame,
            queries: list,
            is_sequential: bool=True, 
            verbose: bool=False
        ):
        answers = []
        if is_sequential: # sequential prompting
            sqlss = [self._generate_sql(
                        query=query, 
                        tbale=table, 
                        template_key=TemplateKey.SQL_GENERATION
                    ) for query in queries]
        else: # parallel prompting
            sqlss = self._generate_sql(
                        query=queries,
                        table=table,
                        template_key=TemplateKey.SQL_GENERATION_2
                    )
            # transpose sqlss
            sqlss = list(map(list, zip(*sqlss)))
        
        for sqls, query in zip(sqlss, queries):
            if verbose: print(f"{query}: {sqls}")

            preds = []
            for sql in sqls:
                try:
                    res = db.execute_query(sql.lower())
                    refined_res = self.parser.parse_sql_result(res)
                    if verbose: print(f"refined: {refined_res}")
                    preds.append(refined_res)
                except Exception as e:
                    continue
            
            top_ans, pred_sqls = majority_vote(
                nsqls=sqls,
                pred_answer_list=preds
            )
            if verbose: print(query, top_ans)
            answers.append(top_ans)

        return answers
    
    @log_decorator
    def reason(self, claim: str, table: pd.DataFrame, verbose=False):
        """
            Reasoning pipeline for CoT
            Input: claim, table
            Output: justification
        """

        def build_dec_prompt(sub_queries: list, answers: list):
            dec_prompt = self.prompter.build_prompt(
                            template_key=TemplateKey.DEC_REASONING_2,
                            table=table,
                            question=query,
                        )
            dec_prompt.extend([{
                "role": "user", 
                "content": "\n".join(sub_queries)
                },{
                "role": "assistant", 
                "content": "\n".join(answers)
                },{
                "role": "user",
                "content": sub_queries[-1]
            }])
            return dec_prompt
            
        db = NeuralDB(
            tables=[table],
            add_row_id=True,
            normalize=False,
            lower_case=True
        )
        # take first query from suggested queries
        suggestions = self._suggest_queries(claim)
        if verbose: print(f"generated queries: {suggestions}")

        justifications = []
        for query in suggestions:
            # decompose queries
            sub_queries = self._decompose_query(query)
            if verbose: print(f"steps of reasoning: {sub_queries}")

            # execute sql corresponding to each subquery
            answers = self._exec_sqls_from_sub_queries(
                            db, table,
                            sub_queries[:-1], 
                            is_sequential=False,
                            verbose=verbose
                        )
            
            sub_queries = [f"Q{i+1}: {query}" for i, query in enumerate(sub_queries)]
            answers = [f"A{i+1}: {str(ans)}" for i, ans in enumerate(answers)]
            # generate prompt for decomposed reasoning
            dec_prompt = build_dec_prompt(sub_queries, answers)
            # if verbose: print(f"full prompt:\n{dec_prompt}")

            answers.extend(self._call_api_2(dec_prompt))
            # print("answers: ", answers)
            justification = self._call_api_2(
                prompt = [
                    {"role": "system", "content": "You are an amazing rhetorician. You are given a sequence of questions and answers that aims to tackle an ultimate question step by step. You need to reframe the sequence to make it look like a coherent, smooth paragraph of logical deduction."},
                    {"role": "user", "content": "\n".join(query + "\n" + answer for query, answer in zip(sub_queries, answers))},
                ]
            )
            justifications.append(justification)

        if verbose: print(f"final justifications: {justifications}")
        return justifications   


if __name__ == "__main__":
    table_reasoner = TableReasoner()
    claim = "US' carbon emission is higher than every North America countries."
    df = pd.read_csv("../Datasets/owid-energy-data.csv")

    # claim = "Africa has the highest population."
    # df = pd.read_csv("../Datasets/owid-energy-data.csv")

    res = table_reasoner._suggest_queries(claim)
    print(res)
