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
from fuzzywuzzy import fuzz
import math

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
        model: Model = Model.GPT3
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
            samples: int = -1,
            model: Model = Model.GPT3
        ):
        """
            Call API using a provide prompt
            Input: prompt
            Output: response
        """
        response = call_model(
            model=model,
            temperature=temperature if temperature > 0 else self.temperature,
            max_decode_steps=self.max_decode_steps,
            prompt=prompt,
            samples=samples if samples > 0 else self.samples
        )

        return response

    def _suggest_queries(self, claim: str, table: pd.DataFrame=None):
        """
            Suggest queries given a claim (and a table)
            Input: @claim
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
            table=table
        )

        queries, vis_tasks, reasons, attributes = self.parser.parse_gen_query(suggestions[0])
    
        # further process the attributes
        for idx, attr in enumerate(attributes):
            # fuzzy match when GPT hallucinating attributes
            if attr not in table.columns: 
                similar_attr = max(table.columns, key=lambda col: fuzz.ratio(col, attr))
                attributes[idx] = similar_attr
            
            # # find the span of each attribute within the claim
            # attr_positions = [(claim.index(attr), claim.index(attr) + len(attr)) for attr in attributes]
            # print(f"Attribute positions: {attr_positions}")
        
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
        # list of list of sqls --> be careful when handling this case
        elif template_key == TemplateKey.SQL_GENERATION_2:
            psqls = [self.parser.parse_sql_2(sql) for sql in sqls]
        
        if fuzzy_match:
            def process_psqls(psqls):
                processed_psqls = []
                for psql in psqls:
                    if isinstance(psql, str):
                        processed_psqls.append(post_process_sql(
                            sql_str=psql, 
                            df=table,
                            process_program_with_fuzzy_match_on_db=fuzzy_match,
                            verbose=False
                        ))
                    elif isinstance(psql, list):
                        processed_psqls.append(process_psqls(psql))
                return processed_psqls
            
            return process_psqls(psqls)
        else:
            return psqls
    
    def _exec_sqls_from_sub_queries(
            self,
            db: NeuralDB,
            queries: list,
            is_sequential: bool=True, 
            verbose: bool=False,
            fuzzy_match: bool=False
        ):
        answers = []
        if is_sequential: # sequential prompting
            sqlss = [self._generate_sql(
                        query=query, 
                        table=db.get_table(), 
                        template_key=TemplateKey.SQL_GENERATION,
                        fuzzy_match=fuzzy_match
                    ) for query in queries]
        else: # parallel prompting
            sqlss = self._generate_sql(
                        query=queries,
                        table=db.get_table(),
                        template_key=TemplateKey.SQL_GENERATION_2,
                        fuzzy_match=fuzzy_match
                    )
            # transpose sqlss
            sqlss = list(map(list, zip(*sqlss)))
        
        def process_ans(ans: list):
            try:
                if len(ans) > 30:
                    # sometimes the answer is too long to fit into the prompt
                    ans = [x for x in ans if not math.isnan(x) and isinstance(x, (int, float))] 
                    return f"Ranging from {str(min(ans))} to {str(max(ans))}"
                return str(ans)
            except Exception as e:
                if verbose: print(e)
                return []
                
        for idx, (sqls, query) in enumerate(zip(sqlss, queries)):
            if verbose: print(f"Q{idx+1}: {query}\nGenerated SQLs: {sqls}")

            preds = []
            for sql in sqls:
                try:
                    res = db.execute_query(sql.lower())
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
            if verbose: print(f"A{idx+1}: {top_ans}\n{'*'*50}")

            answers.append(top_ans)

        return answers
    
    def _evaluate_soundness(self, reasoning:str): 
        evaluation = self._call_api_2(
            prompt = [
                {"role": "system", "content": """You are an amazing logician. You are given a sequence of logical deduction based on real-world data. 
                You need to evaluate the soundness of the reasoning and fix the reasoning while still retain the core idea and be as informative as possible in the following format.
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
    def reason(self, claim: str, table: pd.DataFrame, verbose=False, fuzzy_match=False):
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
        suggestions, vis_tasks, _, attributes = self._suggest_queries(claim, table=db.get_table_df())
                    
        # update table with relevant attributes
        db.update_table(map(lambda x: x.lower(), attributes)) 
        if verbose: 
            print(f"generated queries: {suggestions}")
            if attributes: print(f"mapped attributes: {attributes}")

        reason_map = []
        for idx, query in enumerate(suggestions):
            # decompose queries
            sub_queries = self._decompose_query(query)
            if verbose: print(f"steps of reasoning: {sub_queries}")

            # execute sql corresponding to each subquery (up to the second last one)
            answers = self._exec_sqls_from_sub_queries(
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
            # print("answers: ", answers)
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
            })

        if verbose: print(f"final justifications: {reason_map}")
        
        return {
            "suggestions": reason_map,
            "sub_table": db.get_table_df(),
            "attributes": attributes
        }


if __name__ == "__main__":
    table_reasoner = TableReasoner()
    # claim = "The Phantom is the best movies in term of imdb rating."
    df = pd.read_csv("../Datasets/movies-w-year.csv")
    # table_reasoner.reason(claim, df, verbose=True, fuzzy_match=False)
    db = NeuralDB(
        tables=[df],
        add_row_id=True,
        normalize=False,
        lower_case=True
    )
    # print(db.get_table_df())
    # attributes = ['title', 'imdb rating']
    # db.update_table(attributes)
    # print(db.get_table_df())

    sql = """SELECT content rating FROM w GROUP BY content rating ORDER BY SUM("production budget") DESC LIMIT 1 """
    print(sql)
    psql = post_process_sql(
        sql_str=sql,
        df=db.get_table_df(),
        process_program_with_fuzzy_match_on_db=True,
        verbose=True
    )
    print(psql)
    # print(fuzz.ratio("US", "America"))
    # claim = "Africa has the highest population."
    # df = pd.read_csv("../Datasets/owid-energy-data.csv")

    # res = table_reasoner._suggest_queries(claim, df)
    # print(res)
