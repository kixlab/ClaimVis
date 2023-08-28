# CoT few-shot prompting 
from collections import defaultdict
from functools import cache
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
from rapidfuzz import fuzz
from nl4dv import NL4DV
from nl4dv.utils import helpers
from DataMatching import DataMatcher
import math

class TableReasoner(object):
    def __init__(
            self, 
            temperature=0.0, 
            max_decode_steps=500, 
            samples=1, 
            model=Model.GPT3,
            datamatcher: DataMatcher = None
        ):
        self.prompter = Prompter()
        self.parser = AnsParser()
        self.datamatcher = datamatcher or DataMatcher()

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
        model: Model = Model.GPT3,
        max_decode_steps: int = -1
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
            model=model,
            max_decode_steps=max_decode_steps if max_decode_steps > 0 else self.max_decode_steps,
            temperature=temperature if temperature > 0 else self.temperature,
            prompt=prompt,
            samples=samples if samples > 0 else self.samples
        )

        return prompt, response
        
    def _call_api_2(
            self, 
            prompt: list,
            temperature: float = -1,
            samples: int = -1,
            model: Model = Model.GPT3, 
            max_decode_steps: int = -1
        ):
        """
            Call API using a provide prompt
            Input: prompt
            Output: response
        """
        response = call_model(
            model=model,
            temperature=temperature if temperature > 0 else self.temperature,
            max_decode_steps=max_decode_steps if max_decode_steps > 0 else self.max_decode_steps,
            prompt=prompt,
            samples=samples if samples > 0 else self.samples
        )

        return response

    def _suggest_queries(self, claim: str, table: pd.DataFrame=None, more_attrs: list=None):
        """
            Suggest queries given a claim (and a table)
            Input: @claim, @table
            Output: 
                @query: list of suggested queries
                @vis: a list of vis tasks
                @explain: a list of exlanation for why the queries are suggested
                @attributes: list of attribute used in the queries
        """
        # suggest different queries in form of "[{query: ...}, {query: ...}}]"
        template = TemplateKey.QUERY_GENERATION_2 if table is None \
                                                else TemplateKey.QUERY_GENERATION_3
        _, suggestions = self._call_api_1(
            question=claim,
            template_key=template,
            table=table,
            model=Model.GPT3 # 4
        )
        queries, vis_tasks, reasons, attributes = self.parser.parse_gen_query(suggestions[0])
        attributes, col_set = set(attributes + more_attrs), set(table.columns)
        # add datefield if exist and infer datetime from the queries if not
        time_batch = self.datamatcher.embedder.encode(["time", "date", "year"])
        col_embeds = self.datamatcher.attrs_embeddings[self.datamatcher.datasets.index(table.name)]
        datefields, start_index = [], 1 if table.columns[0] == "row_id" else 0 # row_id added to table so index starts at 1
        for col, embed in zip(table.columns[start_index:], col_embeds): 
            score = self.datamatcher.attr_score_batch(embed, time_batch)
            if score > .5: datefields.append((col, score))
        best_datefield = max(datefields, key=lambda x: x[1])[0] if datefields else None
        attributes.add(best_datefield)
        attributes = list(attributes)

        # infer datetime from the queries if not exist
        if best_datefield:
            oldest_date, newest_date = table[best_datefield].min(), table[best_datefield].max()
            prompt = [
                {"role": "system", "content": f"""Please add date or time to the query if needed. Please use the following default dates: 
                 OLDEST: {oldest_date}
                 NEWEST: {newest_date} 

                 The rule should be as follows (I will give examples for you to follow):
                    1. Add the most recent date when the query lacks date. E.g "US' GDP > China's GDP" --> "US' GDP > China's GDP IN {newest_date}"
                    2. Add the most recent date when the query lacks the end date. E.g "US' GDP > China's GDP SINCE 2010" --> "US' GDP > China's GDP SINCE 2010 TO {newest_date}"
                    3. Add the furthest date when the query lacks the start date. E.g "US' GDP > China's GDP TIL 2010" --> "US' GDP > China's GDP FROM 2010 TIL {oldest_date}"

                    User will give multiple queries inform of:
                    Q1: <query 1>
                    Q2: <query 2>
                    ...
                    Please answer the queries in the following format:
                    A1: <answer 1>
                    A2: <answer 2>
                    ..."""},
                {"role": "user", "content": "\n".join([f"Q{i+1}: {query}" for i, query in enumerate(queries)])}
            ]
            answers = self._call_api_2(prompt, model=Model.GPT4)
            # parse the answers
            queries = self.parser.parse_sql_2(answers[0])

        # further process the attributes
        for idx, attr in reversed(list(enumerate(attributes))):
            # fuzzy match when GPT hallucinating attributes
            if attr not in col_set: 
                similar_attr = max(table.columns, key=lambda col: fuzz.ratio(col, attr))
                if fuzz.ratio(similar_attr, attr) > 50 and similar_attr not in attributes:
                    attributes[idx] = similar_attr
                else: # delete from tail
                    del attributes[idx]       

        # assert every column name in attribute is unique
        assert len(attributes) == len(set(attributes)), "Column names in attributes are not unique"        
        return queries, vis_tasks, reasons, attributes
    
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
            samples: int = 15,
            temperature: float = 0.6,
            fuzzy_match: bool = False,
            max_decode_steps: int = 700
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
                        samples=samples, # need samples to aggregate
                        temperature=temperature, # need some creativity
                        max_decode_steps=max_decode_steps # need to be long enough
                    )

        if template_key == TemplateKey.NSQL_GENERATION:
            psqls = [self.parser.parse_nsql(sql) for sql in sqls]
        elif template_key == TemplateKey.SQL_GENERATION:
            psqls = [self.parser.parse_sql(sql) for sql in sqls]
        # list of list of sqls --> be careful when handling this case
        elif template_key == TemplateKey.SQL_GENERATION_2:
            psqls = [self.parser.parse_sql_2(sql) for sql in sqls]
            # transpose psqls
            psqls = list(map(list, zip(*psqls)))
        
        if fuzzy_match:
            # bottle neck due to fuzzy matching on big tables
            sql_cache, value_map = set(), defaultdict(set) # control the number of distinct sqls
            def process_psqls(psqls):
                processed_psqls = []
                for psql in psqls:
                    if isinstance(psql, str) and psql not in sql_cache:
                        sql_cache.add(psql)
                        new_sql_str, new_val_map = post_process_sql(
                                                        sql_str=psql, 
                                                        df=table,
                                                        matcher=self.datamatcher,
                                                        process_program_with_fuzzy_match_on_db=fuzzy_match,
                                                        verbose=False
                                                    )
                        processed_psqls.append(new_sql_str)
                        for k, v in new_val_map.items():
                            value_map[k].update(v)
                    elif isinstance(psql, list):
                        processed_psqls.append(process_psqls(psql))
                return processed_psqls
            
            return process_psqls(psqls), value_map
        else:
            return psqls, None
    
    def _exec_sqls_from_sub_queries(
            self,
            db: NeuralDB,
            queries: list,
            is_sequential: bool=True, 
            verbose: bool=False,
            fuzzy_match: bool=False
        ):
        answers, value_map = [], None
        if is_sequential: # sequential prompting
            sqlss = [self._generate_sql(
                        query=query, 
                        table=db.get_table(), 
                        template_key=TemplateKey.SQL_GENERATION,
                        fuzzy_match=fuzzy_match
                    ) for query in queries]
        else: # parallel prompting
            sqlss, value_map = self._generate_sql(
                                    query=queries,
                                    table=db.get_table(),
                                    template_key=TemplateKey.SQL_GENERATION_2,
                                    fuzzy_match=fuzzy_match
                                )
        
        def process_ans(ans: list):
            try:
                if len(ans) > 30:
                    # sometimes the answer is too long to fit into the prompt
                    ans = [x for x in ans if isinstance(x, (int, float)) and not math.isnan(x)] 
                    return f"Ranging from {str(min(ans))} to {str(max(ans))}, with average {str(sum(ans)/len(ans))}"
                return str(ans)
            except Exception as e:
                if verbose: print("error with list ans: ", e)
                return []
                
        for idx, (sqls, query) in enumerate(zip(sqlss, queries)):
            if verbose: print(f"Q{idx+1}: {query}\nGenerated SQLs: {sqls}")

            preds = []
            for sql in sqls:
                try:
                    res = db.execute_query(sql)
                    refined_res = self.parser.parse_sql_result(res)
                    # if verbose: print(f"refined: {refined_res}")
                    preds.append(refined_res)
                except Exception as e:
                    continue
            
            top_ans, pred_sqls = majority_vote(
                nsqls=sqls,
                pred_answer_list=preds
            )
            top_ans = process_ans(top_ans)
            if verbose: print(f"A{idx+1}: {top_ans}\n{'*'*75}")

            answers.append(top_ans)

        return answers, value_map
    
    def _evaluate_soundness(self, reasoning:str): 
        evaluation = self._call_api_2(
            prompt = [
                {"role": "system", "content": """You are an amazing logician. You are given a sequence of logical deduction based on real-world data. 
                You need to evaluate the soundness of the reasoning and fix the reasoning while still RETAIN the core idea and be as informative as possible in the following format.
                \{
                    explain: "<TODO: explain why the reasoning is sound or not sound>"   
                    revised: "<TODO: revised reasoning>"
                \}"""},
                {"role": "user", "content": reasoning},
            ],
            model=Model.GPT4 # 4
        )

        return self.parser.parse_evaluation(evaluation[0])
    
    @log_decorator
    def reason(
            self, 
            claim: str, 
            table: pd.DataFrame, 
            verbose=False, 
            fuzzy_match=False,
            more_attrs: list = []):
        """
            Reasoning pipeline for CoT
            Input: claim, table
            Output: justification
        """

        def build_dec_prompt(sub_queries: list, answers: list):
            dec_prompt = self.prompter.build_prompt(
                            template_key=TemplateKey.DEC_REASONING_2,
                            table=db.get_table_df(),
                            question=query,
                        )
            dec_prompt.extend([
                { "role": "user", "content": "\n".join(sub_queries) },
                { "role": "assistant", "content": "\n".join(answers) },
                { "role": "user", "content": sub_queries[-1] }
            ])
            return dec_prompt
            
        db = NeuralDB(
            tables=[table],
            add_row_id=True,
            normalize=False,
            lower_case=False
        )
        # take first query from suggested queries
        suggestions, vis_tasks, _, attributes = self._suggest_queries(
                                                        claim=claim, 
                                                        table=db.get_table_df(), 
                                                        more_attrs=more_attrs
                                                    )
                    
        # update table with relevant attributes
        if attributes: 
            db.update_table(attributes) 
        if verbose: 
            print(f"generated queries: {suggestions}")
            if attributes: print(f"mapped attributes: {attributes}")

        reason_map = []
        for idx, query in enumerate(suggestions):
            # decompose queries
            sub_queries = self._decompose_query(query)
            if verbose: print(f"steps of reasoning: {sub_queries}")

            # execute sql corresponding to each subquery (up to the second last one)
            answers, value_map = self._exec_sqls_from_sub_queries(
                                        db=db,
                                        queries=sub_queries[:-1], 
                                        is_sequential=False,
                                        verbose=verbose,
                                        fuzzy_match=fuzzy_match
                                    )
            sub_queries = [f"Q{i+1}: {query}" for i, query in enumerate(sub_queries)]
            answers = [f"A{i+1}: {ans}" for i, ans in enumerate(answers)]
            # generate prompt for decomposed reasoning
            dec_prompt = build_dec_prompt(sub_queries, answers)
            # if verbose: print(f"full prompt:\n{dec_prompt}")
            answers.extend(self._call_api_2(dec_prompt))

            justification = self._call_api_2(
                prompt = [
                    {"role": "system", "content": """You are an amazing rhetorician. You are given a sequence of questions and answers that aims to tackle an ultimate question step by step. 
                     You need to reframe the sequence to make it look like a coherent, smooth paragraph of logical deduction."""},
                    {"role": "user", "content": "\n".join(query + "\n" + answer for query, answer in zip(sub_queries, answers))},
                ]
            )

            # use GPT4 to evaluate whether the reasoning is sound or not, then revise the reasoning if needed
            evaluation = self._evaluate_soundness(justification[0])
            reason_map.append({
                "query": query,
                "visualization": vis_tasks[idx],
                "reasoning_steps": sub_queries,
                "justification": evaluation,
                "value_map": value_map
            })

        if verbose: print(f"final justifications: {reason_map}\n{'@'*75}")
        
        return {
            "suggestions": reason_map,
            "sub_table": {
                "data": db.get_table_df()
            },
            "attributes": attributes
        }


def main():
    table_reasoner = TableReasoner()
    query = "US' Stock market observes a dump of 30% in the summer of 2008."
    # query = ["What is the total energy consumption of the US in 2012?", "What is the total energy consumption of China in 2012?", "What is the total energy consumption of the world in 2012?"]
    df = pd.read_csv("../Datasets/owid-energy-data.csv")
    table_reasoner.reason(query, df, verbose=True, fuzzy_match=True)
    # table_reasoner._generate_sql(
    #     query=query,
    #     table=df,
    #     template_key=TemplateKey.SQL_GENERATION_2,
    #     fuzzy_match=True
    # )

if __name__ == "__main__":
    pass
    
