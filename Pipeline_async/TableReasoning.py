# CoT few-shot prompting
import sys
import numpy as np

sys.path.append("../Gloc")
sys.path.append("..")

import pandas as pd
from Gloc.generation.claimvis_prompt import Prompter, TemplateKey

# this import also acquires log_decor, don't import it again
from Gloc.utils.async_llm import *
from Gloc.processor.ans_parser import AnsParser
from Gloc.utils.normalizer import post_process_sql
from Gloc.nsql.database import NeuralDB
from Gloc.utils.utils import majority_vote
from rapidfuzz import fuzz
from DataMatching import DataMatcher
from pyinstrument import Profiler
from models import UserClaimBody, ClaimMap, Dataset
from collections import defaultdict
from functools import cache
from itertools import zip_longest
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
from Gloc.utils.normalizer import _get_matched_cells
import json
import random
import re
import spacy
import math


class TableReasoner(object):
    MIN_DATE = 1960
    MAX_DATE = 2020
    INDICATOR = {
        "alternative + complementary metrics": "value",
        "years": "datetime",
        "countries": "country",
    }

    RECOMMENDATION_PROMPTS = {
        "alternative + complementary metrics": [
            {
                "role": "system",
                "content": """For a given statement, a context paragraph, and the background of the reader, please suggest sets of statistical indicators to contextualize the statement. Select the indicators in the following order, but skip if there is no appropriate country. 
a) indicators referred in the context, b) indicators breaking down the statement, c) complementary indicators with potential correlation with the statement, d) alternative indicators to investigate the statement.
Provide a one-sentence teaser question for each set of values to intrigue the reader to explore the context. Please wrap each value with @() if it does not directly refer to the specific value. Respond as JSON in the following format:

[{
    "indicator": "",
    "sets": [{
        "values": ["<value 11>", "<value 12>", ...],
        "teaser": "<teaser1>"
    }, 
    {
        "values": ["<value 21>", "<value 22>", ...],
        "teaser": "<teaser2>"
    }]
}, ...]

Each suggested set of value must be CONSISTENT with the indicator!""",
            },
            {
                "role": "user",
                "content": """Reader's background: South Korea

Context: A significant amount of New Zealand's GDP comes from tourism, and our GDP overall is a lot lower than in the states, so in reality we should have been financially impacted more not less, and have less money to fight the virus than the US did.
Statement: A significant amount of New Zealand's GDP comes from tourism""",
            },
            {
                "role": "assistant",
                "content": """[{
  "indicator": "share of GDP from tourism",
  "sets": [{
    "values": ["GDP"],
    "teaser": "What's the total GDP of New Zealand?"
  }, {
    "values": ["exports", "international trade"],
    "teaser": "How does New Zealand's reliance on tourism compare to its reliance on exports or international trade?"
  }, 
  {
    "values": ["agriculture", "manufacturing"],
    "teaser": "What other sectors contribute significantly to New Zealand's GDP besides tourism?"
  }]
}]""",
            },
            {"role": "user", "content": """"""},
            {"role": "assistant", "content": """"""},
        ],
        "years": [
            {
                "role": "system",
                "content": """For a given statement and a context paragraph, please suggest different sets of years to compare the statement in other contexts. Only select years before 2023. Select the years in the following order, but skip if there is no appropriate country. 
a) time period referred in the context, b) time period related to reader's background,  c) time period related or similar to the period in the statement, d) time period similar or related to reader's background, e) time periods with extreme values in terms of the statement's context.
Provide a one-sentence teaser question for each set of values to intrigue the reader to explore the context. Please wrap each value with @() if it does not directly refer to the specific value. Respond as JSON in the following format:
    [{
        "values": ["<value 11>", "<value 12>", ...],
        "teaser": "<teaser1>"
        }, {
        "values": ["<value 21>", "<value 22>", ...],
        "teaser": "<teaser2>"
    }]

Each suggested set of values must be CONSISTENT with the indicator!""",
            },
            {
                "role": "user",
                "content": """Reader's background: South Korea

Context: A significant amount of New Zealand's GDP comes from tourism, and our GDP overall is a lot lower than in the states, so in reality we should have been financially impacted more not less, and have less money to fight the virus than the US did.
Statement: A significant amount of New Zealand's GDP comes from tourism""",
            },
            {
                "role": "assistant",
                "content": """[{
    "values": ["2019", "2021"],
    "teaser": "What was the impact of the pandemic on New Zealand's tourism industry in 2020?"
},
{
    "values": ["@(Year with the largest proportion of tourism in their GDP)"],
    "teaser": "When did New Zealand have the largest proportion of tourism in their GDP?"
}]""",
            },
        ],
        "countries": [{
            "role": "system",
            "content": """For a given statement, a context paragraph, and the background of the reader, please suggest different sets of countries to compare the statement in other contexts. Select the countries in following order, but skip if there is no appropriate country. 
a) countries referred in the context, b) countries of reader's background,  c) countries related or similar to the country in the statement, d) countries similar or related to reader's background, e) countries extreme in terms of the statement's context. Provide a one-sentence teaser question for each set of values to intrigue the reader to explore the context. Wrap each value with @() if it does not directly refer to the specific value. Respond as JSON in the following format:
    [{
        "values": ["<value 11>", "<value 12>", ...],
        "teaser": "<teaser1>"
        }, {
        "values": ["<value 21>", "<value 22>", ...],
        "teaser": "<teaser2>"
    }]

Each suggested set of values must be CONSISTENT with the indicator!"""
        },
        {
            "role": "user",
            "content": """Reader's background: South Korea

Context: A significant amount of New Zealand's GDP comes from tourism, and our GDP overall is a lot lower than in the states, so in reality we should have been financially impacted more not less, and have less money to fight the virus than the US did.
Statement: A significant amount of New Zealand's GDP comes from tourism"""
        },
        {
            "role": "assistant",
            "content": """[{
    "values": ["South Korea"],
    "teaser": "How does the impact of tourism on New Zealand's economy compare to that of the reader's country?"
},{
    "values": ["Thailand", "Greece", "Spain"],
    "teaser": "How is the impact of tourism on the economy in other countries known for tourism?"
}, {
    "values": ["Australia", "Canada", "Norway"],
    "teaser": "How is the impact of tourism on the economy in other developed countries similar to New Zealand?"
}, {
    "values": ["Japan", "China", "Taiwan"],
    "teaser": "How does the impact of tourism on New Zealand's economy compare to that of other East Asian countries similar to South Korea?"
}, {
    "values": ["@(Top 3 countries with the highest contribution to GDP from tourism)"],
    "teaser": "What are the top 3 countries with the highest contribution to GDP from tourism?"
}]"""
        }],
    }

    def __init__(
        self,
        temperature=0.0,
        max_decode_steps=300,
        samples=1,
        model=Model.GPT3,
        datamatcher: DataMatcher = None,
    ):
        self.prompter = Prompter()
        self.parser = AnsParser()
        self.dm = datamatcher

        self.temperature = temperature  # to change
        self.max_decode_steps = max_decode_steps  # fixed
        self.samples = samples  # to change
        self.model = model  # fixed
        self.nlp = None

        self.date_pattern = r"(@\(.*?\)|\d{4})(\s*-\s*(@\(.*?\)|\d{4}))?"

    async def _call_api_1(
        self: object,
        question: str,
        template_key: TemplateKey,
        table: pd.DataFrame = None,
        samples: int = -1,
        temperature: float = -1,
        model: Model = Model.GPT3,
        max_decode_steps: int = -1,
    ):
        """
        Call API for few-shot prompting using a question, a template, and a table
        Input: question, template_key, table
        Output: prompt, response
        """
        prompt = self.prompter.build_prompt(
            template_key=template_key, table=table, question=question
        )

        response = await call_model(
            model=model,
            max_decode_steps=max_decode_steps
            if max_decode_steps > 0
            else self.max_decode_steps,
            temperature=temperature if temperature > 0 else self.temperature,
            prompt=prompt,
            samples=samples if samples > 0 else self.samples,
        )

        return prompt, response

    async def _call_api_2(
        self,
        prompt: list,
        temperature: float = -1,
        samples: int = -1,
        model: Model = Model.GPT3,
        max_decode_steps: int = -1,
    ):
        """
        Call API using a provide prompt
        Input: prompt
        Output: response
        """
        response = await call_model(
            model=model,
            temperature=temperature if temperature > 0 else self.temperature,
            max_decode_steps=max_decode_steps
            if max_decode_steps > 0
            else self.max_decode_steps,
            prompt=prompt,
            samples=samples if samples > 0 else self.samples,
        )

        return response

    # @log_decorator
    async def _tag_claim(
        self,
        claim: str,
        template_key: TemplateKey = TemplateKey.CLAIM_TAGGING,
        model: Model = Model.GPT3,
        verbose: bool = False,
        fewshot_samples: int = 6,
        gen_samples: int = 1,
    ):
        """
        Tag claim with claim type
        Input: claim
        Output: claim type
        """
        if model in [Model.GPT_TAG_2, Model.GPT_TAG_3, Model.GPT_TAG_4, Model.GPT4]:
            msg = [
                {
                    "role": "system",
                    "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible.""",
                },
            ]
            filedir = "../Gloc/generation/finetune/claim_tagging.jsonl"
            with open(filedir, "r") as file:
                data = [json.loads(line) for line in file]

                # indices = list(range(0, len(data)))
                # sampled_indices = [random.sample(indices, samples)]
                sampled_indices = range(0, fewshot_samples)
                random_sample = [
                    msg for idx in sampled_indices for msg in data[idx]["messages"][1:]
                ]

                msg.extend(random_sample)
                
            msg.append({"role": "user", "content": claim})
            claim_tag = await self._call_api_2(msg, model=model, max_decode_steps=700, samples=gen_samples, temperature=.0)
        else:
            _, claim_tag = await self._call_api_1(
                                question=claim,
                                template_key=template_key,
                                model=model
                            )

        data = [json.loads(sample) for sample in claim_tag]
        # if verbose:
        #     print(f"model: {model}")
        #     for i, item in enumerate(data, 1):
        #         print(f"claim tag {i}: {item}")
        # majority vote
        if len(data) > 1:
            # <TODO: Majority vote>
            answer = None
        else:
            answer = data[0]
        # check if the format of the response is correct
        for idx, country in enumerate(answer["country"]):
            if country.startswith("@(") and country not in answer["vis"]:
                if country[:-2] in answer["vis"]: # lack ?
                    answer["country"][idx] = country[:-2] + "?)"

        return answer
        
    async def _infer_country(
            self, claim: str, 
            dates: list[str], 
            values: list[str], 
            table: pd.DataFrame,
            verbose: bool=True,
        ):
        queries, combos= set(), list(product(dates, values))
        for date, value in combos:
            if '-' in date:
                start, end = date.split('-')
                date = "from {} to {}".format(start, end)
            else:
                date = "in {}".format(date)
            
            if claim.startswith("Year"):
                queries.add("What is the {}?".format(claim))
                break # no need to iterate through all combos
            elif not re.match(r'(Top|Bottom) (\d+) countries$', claim): # no need to add value
                queries.add("What are the {} {}?".format(claim, date))
            else: 
                queries.add("What are the {} of {} {}?".format(claim, value, date))

        db = NeuralDB([table], add_row_id=False, normalize=False, lower_case=False)

        if verbose: print(f"queries: {queries}")
        response = await self._exec_sqls_from_sub_queries(
            db, queries, fuzzy_match=True, verbose=verbose
        )

        try:
            def safe_eval(items:str):
                items = re.sub(r'\bnan\b', 'float("nan")', items)
                return eval(items)
            countries = [item for sublist in response[0] for item in safe_eval(sublist[0])]
            if claim.startswith("Year"):
                return set(item for item in countries if isinstance(item, int))
            else:
                return set(item for item in countries if isinstance(item, str))
        except Exception as e:
            if verbose: print("Error:", e)
            return []

    async def _infer_datetime(
        self,
        queries: list,
        start_default: int = 1960,
        end_default: int = 2021,
        use_llm: bool = False,
    ):
        """
        Infer datetime from queries
        Input: queries
        Output: queries with datetime
        """
        if use_llm:
            prompt = [
                {
                    "role": "system",
                    "content": f"""Please add date or time to the query if needed. 
                 
                 The rules will be demonstrated via the following examples. Let's use default oldest and newest dates as 1960 and 2021, respectively.
                    1. Add the most recent date when the query lacks date. E.g "US' GDP > China's GDP" --> "US' GDP > China's GDP in 2021"
                    2. Add the most recent date when the query lacks the end date. E.g "US' GDP > China's GDP since 2010" --> "US' GDP > China's GDP since 2010 to 2021"
                    3. Add the oldest date when the query lacks the start date. E.g "US' GDP > China's GDP until 2010" --> "US' GDP > China's GDP from 1960 until 2010"
                    4. Do not add date when the query already doesn't need more date. E.g "US' GDP > China's GDP in 2010" --> "US' GDP > China's GDP in 2010"

                    User will give multiple queries inform of
                 /*
                 Please use the following default dates:
                 OLDEST: <oldest date>
                 NEWEST: <newest date>
                    Q1: <query 1>
                    Q2: <query 2>
                    ...
                 */

                    Please answer the queries in the following format:
                    A1: <answer 1>
                    A2: <answer 2>
                    ...""",
                },
                {
                    "role": "user",
                    "content": f"""Please use the following default dates: 
                 OLDEST: {start_default}
                 NEWEST: {end_default}"""
                    + "\n".join(
                        [f"Q{i+1}: {query}" for i, query in enumerate(queries)]
                    ),
                },
            ]
            answers = await self._call_api_2(prompt, model=Model.GPT4)
            # parse the answers
            answers = self.parser.parse_sql_2(answers[0])
            return answers
        else:
            self.nlp = spacy.load("en_core_web_sm")
            # use spacy dependency parsing
            for idx, query in enumerate(queries):
                doc, dates = self.nlp(query), []
                # count the number of dates within the query
                for ind, token in enumerate(doc):
                    if token.ent_type_ == "DATE" and token.head.pos_ in [
                        "ADP",
                        "SCONJ",
                    ]:
                        dates.append(ind)

                if len(dates) == 0:  # add the most recent date
                    query = f"In {end_default}, {query}"
                elif len(dates) == 1:  # rules
                    ind = dates[0] - 1
                    if doc[ind].text.lower() in ["since", "from"]:
                        query = f"{doc[:ind]} from {doc[ind+1]} to {end_default} {doc[ind+2:]}"
                    elif doc[ind].text.lower() in ["til", "until"]:
                        query = f"{doc[:ind]} from {start_default} {doc[ind:]}"

                queries[idx] = query

            return queries

    async def _suggest_queries(
        self, claim: str, table: pd.DataFrame = None, more_attrs: list = None
    ):
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
        template = (
            TemplateKey.QUERY_GENERATION_2
            if table is None
            else TemplateKey.QUERY_GENERATION_3
        )
        _, suggestions = await self._call_api_1(
            question=claim, template_key=template, table=table, model=Model.GPT3  # 4
        )
        queries, vis_tasks, reasons, attributes = self.parser.parse_gen_query(
            suggestions[0]
        )

        attributes, col_set = set(attributes + more_attrs), set(table.columns)
        # add datefield if exist and infer datetime from the queries if needed
        time_batch = self.dm.encode(["time", "date", "year"])
        col_embeds = list(self.dm.attrs_embeddings[table.name].values())
        datefields, start_index = (
            [],
            1 if table.columns[0] == "row_id" else 0,
        )  # row_id added to table so index starts at 1
        for col, embed in zip(table.columns[start_index:], col_embeds):
            score = self.dm.attr_score_batch(embed, time_batch)
            if score > 0.5:
                datefields.append((col, score))
        best_datefield = max(datefields, key=lambda x: x[1])[0] if datefields else None
        # print("best datefield: ", best_datefield)

        if best_datefield:  # infer datetime
            oldest_date, newest_date = (
                table[best_datefield].min(),
                2020,
            )  # 2020 has the most data
            answers = await self._infer_datetime(
                queries + vis_tasks, oldest_date, newest_date, use_llm=True
            )
            # chop to queries and vis tasks
            queries, vis_tasks = answers[: len(queries)], answers[len(queries) :]

        attributes.add(best_datefield)
        attributes = list(attributes)
        # further process the attributes
        for idx, attr in reversed(list(enumerate(attributes))):
            # fuzzy match when GPT hallucinating attributes
            if attr not in col_set:
                lower_attr = attr.lower()
                similar_attr = max(
                    table.columns, key=lambda col: fuzz.ratio(col, lower_attr)
                )
                if (
                    fuzz.ratio(similar_attr.lower(), lower_attr) > 50
                    and similar_attr not in attributes
                ):
                    attributes[idx] = similar_attr
                else:  # delete from tail
                    del attributes[idx]

        # assert every column name in attribute is unique
        assert len(attributes) == len(set(attributes)), "Column names in attributes are not unique"     
        return queries, vis_tasks, reasons, attributes
    
    def create_recommendation_prompt(self, variable: str, userClaim: str, paragraph: str, userBackground: str = 'South Korea'):
        prompt = list(self.RECOMMENDATION_PROMPTS[variable])

        final_prompt = {
            "role": "user",
            "content": f"""Reader's background: {userBackground} \nContext: {paragraph}\nStatement: "{userClaim}" """,
        }

        prompt.append(final_prompt)

        return prompt
    
    async def _pairwise_comparison(self, claim: UserClaimBody, questionA, questionB, cut_context):
        """
        Compare two questions and return the one with higher priority
        Input: questionA, questionB
        Output: question with higher priority
        """
        # <TODO>
        A = questionA
        B = questionB
        if random.random() < 0.5:
            A = questionB
            B = questionA

        prompt = [
            {
                "role": "system",
                "content": """Given a claim, surrounding paragraph, a reader's context, and a pair of questions to contextualize the claim, give a judgment on which is better in terms of the following criteria:

1. Is the question testable with quantitative data? 
2. Is the question interesting for the reader to explore?
3. Is the question helpful to consider the broader context around the claim?
4. Is the question suggesting a new aspect?

Simply ANSWER A, B, or SAME
"""
            },
            {
                "role": "user",
                "content": f"""Reader's background: {claim.context} \nParagraph: {cut_context}\nClaim: "{claim.userClaim}"\nQuestion A: {A}\nQuestion B: {B}"""
            }
        ]
        response = await self._call_api_2(prompt, model=Model.GPT4, temperature=0.25, max_decode_steps=1)

        if response[0] == "A":
            return questionA
        elif response[0] == "B":
            return questionB
        elif response[0] == "SAME":
            return None
        else:
            print(response[0])
            return 0
        
    async def _fill_each_element(self, questions: list, i, j, claim: UserClaimBody, matrix: np.ndarray, cut_context: str, verbose=True):
        """
        Fill each element in the question
        Input: questions, claim
        Output: questions with filled elements
        """
        # <TODO>
        questionA = questions[i]
        questionB = questions[j]
        question = await self._pairwise_comparison(claim, questionA, questionB, cut_context)
        if question == questionA:
            matrix[i][j] += 1
        elif question == questionB:
            matrix[j][i] += 1
        elif question == None:
            matrix[i][j] += 0.5
            matrix[j][i] += 0.5
        return 
    
    def cut_context(self, context: str, userClaim: str) -> str:
        """
        Cut the context into sentences using spacy
        Input: context
        Output: sentences
        """
        # <TODO>
        self.nlp = spacy.load("en_core_web_sm") if self.nlp is None else self.nlp
        doc = self.nlp(context)
        sentences = [sent.text for sent in doc.sents]
        claim = userClaim.strip()
        ## if there are more than three sentences, leave the ones with the claim, the one before the claim, and the one after the claim
        if len(sentences) > 3:
            ## Find the sentence containing the claim
            claim_index = -1
            for i in range(len(sentences)):
                if claim in sentences[i]:
                    claim_index = i
                    break

            sentences = sentences[max(0, claim_index - 1): min(claim_index + 2, len(sentences))]
            ## join sentences into a string
            sentences = " ".join(sentences)
            return sentences
        else:
            return context

    async def _bubble_sort(self, questions: list, claim: UserClaimBody, verbose=True):
        """
        Sort questions based on the claim
        Input: questions, claim
        Output: sorted questions
        """
        # <TODO>
        ## First, mix the order of questions to avoid bias  
        random.shuffle(questions)

        ## Then, compute the matrix of pairwise comparison for each question
        matrix = np.zeros((len(questions), len(questions)))
        cut_context = self.cut_context(claim.paragraph, claim.userClaim)

        result = await asyncio.gather(*[self._fill_each_element(questions, i, j, claim, matrix, cut_context, verbose) for i in range(len(questions)) for j in range(len(questions)) if i > j])

        ## From the matrix, compute the score of each question
        if verbose: print(f"score matrix: {matrix}")
        score = np.sum(matrix, axis=1)

        ## Then, sort the questions so higher priority questions are at the top, from bottom to the top
        for i in range(len(questions)):
            questions[i]["score"] = score[i]

        questions = sorted(questions, key=lambda x: x['score'], reverse=True)
        
        ## Then, sort the questions so higher priority questions are at the top, from bottom to the top
        # for i in range(3): ## Maybe just present 3 questions
        #     for j in range(len(questions) - i - 1, 0, -1):
        #         questionA = questions[j - 1]
        #         questionB = questions[j]
        #         question = await self._pairwise_comparison(claim, questionA, questionB)
        #         if question == questionB:
        #             questions[j - 1], questions[j] = questions[j], questions[j - 1]

        
        for i in range(len(questions)):
            questions[i]["rank"] = i + 1

        return questions

    
#     async def _rank_suggestions(self, suggestions: list, claim: UserClaimBody, verbose: bool=True):
#         """
#         Rank suggestions based on the claim
#         Input: suggestions, claim
#         Output: ranked suggestions
#         """
#         # <TODO>
    
#         prompt = [
#             {
#                 "role": "system",
#                 "content": """Given a claim, surrounding paragraph, a reader's context, and a list of questions to contextualize the claim in JSON format, give a single ranking of the recommendations. When ranking, consider the following:

# 1. Is the question interesting for the reader to explore?
# 2. Is the question helpful to consider the broader context and global trend around the claim?
# 3. Is the question suggesting a new aspect?

# Give the ranking in the following JSON format.
# [
# {
#    rank: 1,
#    question: <question>
# },
# ...
# ]"""
#             },
#             {
#                 "role": "user",
#                 "content": f"""Reader's background: {claim.context} \nParagraph: {claim.paragraph}\nClaim: "{claim.userClaim}"\nSuggestions: {suggestions}"""
#             }
#         ]

#         response = await self._call_api_2(prompt, model=Model.GPT4, temperature=0.25, max_decode_steps=600)
#         if verbose: print(f"response: {response}")
#         temp_res = json.loads(response[0])
#         for s in suggestions:
#             for r in temp_res:
#                 if s['explain'] == r['question']:
#                     s['rank'] = r['rank']
#                     break
#         suggestions = sorted(suggestions, key=lambda x: x['rank'])
#         return suggestions

    async def _suggest_variable(self, claim: UserClaimBody, variable: str, verbose: bool=True):
        prompt = self.create_recommendation_prompt(variable, claim.userClaim, claim.paragraph, userBackground=claim.context)
        
        response = await self._call_api_2(
            prompt, model=Model.GPT3, temperature=0.8, max_decode_steps=600
        )
        # if verbose: print(f"response: {response}")
        temp_res = json.loads(response[0])
        if variable == 'alternative + complementary metrics':
            res = [rr for r in temp_res for rr in r['sets']] 
        else:
            res = temp_res
    
        new_res = list(
            map(
                lambda x: {
                    "field": self.INDICATOR[variable],
                    "values": [str(val) for val in x["values"]],
                    "explain": x["teaser"],
                },
                res,
            )
        )
        return new_res
    
    async def _suggest_exploration(self, claim: UserClaimBody, verbose: bool=True):
        prompt = [
            {"role": "system", "content": """You are a keen learner. You are given a statement, and a context paragraph. Please suggest a list of questions that could explore the context of the statement. Provide a one-sentence explanation for each question and respond as JSON in following format:
            [{
                "question": "<question 1>",
                "explain": "<explanation>"
                },{
                "question": "<question 2>",
                "explain": "<explanation>"
            },...]"""},
            {"role": "user", "content": f"""Context: {claim.paragraph}\nStatement: "{claim.userClaim}" """}
        ]
        response = await self._call_api_2(prompt, model=Model.GPT3, temperature=.8)
        # if verbose: print(f"response: {response}")
        return json.loads(response[0])
    
    async def _suggest_queries_2(self, body: UserClaimBody, verbose: bool=True, model: Model=Model.GPT_TAG_4):
        tasks = [self._suggest_variable(body, ind, verbose=verbose) \
                               for ind in self.INDICATOR] \
                + [self._tag_claim(
                    body.userClaim, TemplateKey.CLAIM_TAGGING_2, 
                      model=model, verbose=verbose, fewshot_samples=10
                )]
        attributes, years, countries, claim_tag = await asyncio.gather(*tasks)

        # ranked_suggestions = self._rank_suggestions(attributes, body, verbose=verbose)
        # run rank_suggestions concurrent with the other logic
        loop = asyncio.get_event_loop()
        ranked_suggestions = asyncio.create_task(
            self._bubble_sort((attributes + years + countries), body, verbose=verbose)
        )
        variables, claim_tag["cloze_vis"] = {
            "X": self.MIN_DATE,
            "Y": self.MAX_DATE,
        }, claim_tag["vis"]
        print(f"claim_tag: {claim_tag}")
        for idx, tagged_date in enumerate(claim_tag["datetime"]):
            print(f"tagged_date: {tagged_date}")
            match = re.search(self.date_pattern, tagged_date)
            start_end = (
                [match.group(1), match.group(3)] if match.group(3) else [match.group(1)]
            )
            for date in start_end:
                if date.startswith("@("):
                    val = eval(date[2:-1], variables)
                    claim_tag["vis"] = claim_tag["vis"].replace(
                        f"{{{date}}}", f"{{{str(val)}}}"
                    )
                    claim_tag["datetime"][idx] = claim_tag["datetime"][idx].replace(
                        f"{date}", f"{str(val)}"
                    )
                else:
                    val = date

                claim_tag["rephrase"] = claim_tag["rephrase"].replace(f"{{{date}}}", f"{str(val)}")
                claim_tag["cloze_vis"] = claim_tag["cloze_vis"].replace(f"{{{date}}}", "{date}")

        for idx, tagged_country in enumerate(claim_tag["country"]):
            claim_tag["cloze_vis"] = claim_tag["cloze_vis"].replace(f"{{{tagged_country}}}", "{country}")
            if tagged_country.startswith("@("):
                if all(p not in tagged_country.lower() for p in ["countries", "country"]):
                    new_tagged_country = f"@(Countries of {tagged_country[2:]}"
                    claim_tag["vis"] = claim_tag["vis"].replace(f"{{{tagged_country}}}", f"{{{new_tagged_country}}}")
                    claim_tag["country"][idx] = new_tagged_country
                else:
                    new_tagged_country = tagged_country
                claim_tag["rephrase"] = claim_tag["rephrase"].replace(f"{{{tagged_country}}}", f"{new_tagged_country[2:-2]}") 
            
        for tagged_attr in claim_tag["value"]:
            claim_tag["cloze_vis"] = claim_tag["cloze_vis"].replace(f"{{{tagged_attr['rephrase']}}}", "{value}")
            claim_tag['rephrase'] = claim_tag['rephrase'].replace(f"{{{tagged_attr['rephrase']}}}", f"{tagged_attr['rephrase']}") 
        
        rank_result = await ranked_suggestions
        if verbose: print(f"rank_result: {rank_result}")
        claim_tag["suggestion"] = rank_result # attributes + years + countries
        claim_tag["mapping"] = dict()
        claim_tag["value"] = [attr["rephrase"] for attr in claim_tag["value"]]
        claim_tag["date"] = claim_tag["datetime"]
        del claim_tag["datetime"]
        # if verbose: print(f"claim tag: {claim_tag}\n{'@'*75}")
        return ClaimMap(**claim_tag)

    async def _get_relevant_datasets(self, claim_map: ClaimMap, verbose: bool=True):
        """
		1. Infer the most related attributes
		2. Infer the @() countries
		3. Infer @() years????
	"""
        value_keywords = [keyword for sublist in claim_map.suggestion for keyword in sublist.values if sublist.field == "value" or keyword.startswith("@(")]
        country_keywords = [keyword[2:-2].replace("Country", "").replace("Countries", "").strip() for keyword in claim_map.country if keyword.startswith("@(")]
        keywords = country_keywords + claim_map.value + value_keywords
        print("keywords:", keywords)
        top_k_datasets = await self.dm.find_top_k_datasets("", k=5, method="gpt", verbose=verbose, keywords=keywords)
        datasets = [Dataset(name=name, description=description, score=score, fields=fields) 
            for name, description, score, fields in top_k_datasets]

        # 1. Infer the most related attributes
        table, country_attr, date_attr, fields, embeddings, _ = self.dm.merge_datasets(datasets)
        attributes = claim_map.value + [keyword for keyword in value_keywords if not keyword.startswith("@(")]
        scores = cosine_similarity(self.dm.encode(attributes), embeddings)
        argmax_indices = scores.argmax(axis=1)
        new_attributes = [fields[i] for i in argmax_indices] 
        
        warn_flag, warning = False, ""
        for i, score in enumerate(scores):
            if score[argmax_indices[i]] < 0.5:
                if i < len(claim_map.value):
                    warning = f"The pipeline is not confident with {attributes[i]}."
                    print(f"{'@'*100}\n{warning}. Score: {score[argmax_indices[i]]}\n{'@'*100}")
                    warn_flag = True
                else:
                    rec_warn = f"The pipeline is not confident about the suggested value attribute {attributes[i]}"
                    print(f"{'@'*100}\n{rec_warn}. Score: {score[argmax_indices[i]]}\n{'@'*100}")
                    new_attributes[i] = "!" + new_attributes[i]

        if not warn_flag:
            print(f"{'@'*100}\nThe pipeline is confident. Score: {min(score[argmax_indices[i]] for i, score in enumerate(scores[:len(claim_map.value)]))}\n{'@'*100}")

        print("new_attributes:", new_attributes)
        claim_map.mapping.update({attr: new_attributes[i] for i, attr in enumerate(attributes)})
        new_attributes = new_attributes[:len(claim_map.value)] # only update the value attributes

        # update date and country real attribute name
        print("Country:", country_attr, "Date:", date_attr)
        claim_map.mapping.update({"date": date_attr, "country": country_attr})
        claim_map.cloze_vis = claim_map.cloze_vis.replace("{date}", f'{{{date_attr}}}').replace("{country}", f'{{{country_attr}}}')

        # 2. Infer the @() countries/ @() years from both the claim and the suggested values
        infer_country_tasks, country_to_infer = [], []
        for idx, country in enumerate(claim_map.country):
            if country.startswith('@('):
                if any(p in country for p in ["Bottom", "Top", "with", "Countries of"]):
                    infer_country_tasks.append(
                        self._infer_country(
                            country[2:-2], 
                            claim_map.date, 
                            new_attributes, table
                        )
                    )	
                    country_to_infer.append(country)
                else: # query like @(Asian countries?) have been handled by the _suggest_variable module
                    cntry_sets = [cntry_set for cntry_set in claim_map.suggestion if cntry_set.field == self.INDICATOR["countries"]]
                    suggest_countries = set(cntry for sublist in cntry_sets for cntry in sublist.values)
                    actual_suggest_countries = []
                    for cntry in suggest_countries:
                        matched_cells = _get_matched_cells(cntry, self.dm, table, attr=country_attr)
                        if matched_cells:
                            actual_suggest_countries.append(matched_cells[0][0])
                    # suggest_countries = random.sample(suggest_countries, 5)
                    claim_map.mapping[country] = actual_suggest_countries[:5] # take the top 5 suggested
            else:
                matched_cells = _get_matched_cells(country, self.dm, table, attr=country_attr)
                if matched_cells:
                    claim_map.mapping[country] = matched_cells[0][0]
                else:
                    claim_map.mapping[country] = ""

        for suggest in claim_map.suggestion: 
            for val in suggest.values:
                if val.startswith('@('):
                    infer_country_tasks.append(
                        self._infer_country(
                            val[2:-2 if val[-2]=="?" else -1], claim_map.date, 
                            new_attributes, table
                        )
                    )
                    country_to_infer.append(val)

        inferred_countries = await asyncio.gather(*infer_country_tasks)
        claim_map.mapping.update({country_to_infer[idx]: country_list for idx, country_list in enumerate(inferred_countries)})

        for suggest in claim_map.suggestion: 
            for idx in reversed(range(len(suggest.values))):
                val = suggest.values[idx]
                if (val.startswith('@(') or suggest.field == "value"):
                    if not claim_map.mapping[val]:
                        del suggest.values[idx]
                    elif isinstance(claim_map.mapping[val], str) and claim_map.mapping[val].startswith("!"):
                        suggest.caution.append(val)
                        claim_map.mapping[val] = claim_map.mapping[val][1:]
                        del suggest.values[idx]

        return {
            "datasets": datasets,
            "claim_map": claim_map,
            "warning": warning
        }

    async def _decompose_query(self, query: str):
        """
        Decompose query into subqueries
        Input: query
        Output: list of subqueries
        """
        _, decomposed_ans = await self._call_api_1(
            question=query, template_key=TemplateKey.QUERY_DECOMPOSE
        )

        return self.parser.parse_dec_reasoning(decomposed_ans[0])

    async def _decompose_colunns(self, claim: str, table: pd.DataFrame):
        """
        Decompose table into subtable using column decomposition
        Input: claim, table
        Output: subtable
        """
        _, decomposed_cols = await self._call_api_1(
            question=claim, template_key=TemplateKey.COL_DECOMPOSE, table=table
        )

        cols = self.parser.parse_col_dec(decomposed_cols[0])
        return table.loc[:, cols]

    async def _generate_sql(
        self,
        query: str,
        table: pd.DataFrame,
        template_key: TemplateKey,
        samples: int = 15,
        temperature: float = 0.6,
        fuzzy_match: bool = False,
        max_decode_steps: int = 700,
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

        if template_key not in [
            TemplateKey.NSQL_GENERATION,
            TemplateKey.SQL_GENERATION,
            TemplateKey.SQL_GENERATION_2,
        ]:
            raise ValueError("Invalid template key for SQL generation")

        _, sqls = await self._call_api_1(
            question=query,
            template_key=template_key,
            table=table,
            samples=samples,  # need samples to aggregate
            temperature=temperature,  # need some creativity
            max_decode_steps=max_decode_steps,  # need to be long enough
        )

        if template_key == TemplateKey.NSQL_GENERATION:
            psqls = [self.parser.parse_nsql(sql) for sql in sqls]
        elif template_key == TemplateKey.SQL_GENERATION:
            psqls = [self.parser.parse_sql(sql) for sql in sqls]
        # list of list of sqls --> be careful when handling this case
        elif template_key == TemplateKey.SQL_GENERATION_2:
            psqls = [self.parser.parse_sql_2(sql) for sql in sqls]
            # transpose psqls, pad with "SELECT" if needed
            psqls = list(map(list, zip_longest(*psqls, fillvalue="SELECT")))

        if fuzzy_match:
            # bottle neck due to fuzzy matching on big tables
            sql_cache, value_map = set(), defaultdict(
                set
            )  # control the number of distinct sqls

            def process_psqls(psqls):
                processed_psqls = []
                for psql in psqls:
                    if isinstance(psql, str) and psql not in sql_cache:
                        sql_cache.add(psql)
                        new_sql_str, new_val_map = post_process_sql(
                            sql_str=psql,
                            df=table,
                            matcher=self.dm,
                            process_program_with_fuzzy_match_on_db=fuzzy_match,
                            verbose=False,
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

    async def _exec_sqls_from_sub_queries(
        self,
        db: NeuralDB,
        queries: list,
        is_sequential: bool = False,
        verbose: bool = False,
        fuzzy_match: bool = False,
    ):
        answers, value_map = [], None
        if is_sequential:  # sequential prompting
            sqlss = [
                await self._generate_sql(
                    query=query,
                    table=db.get_table(),
                    template_key=TemplateKey.SQL_GENERATION,
                    fuzzy_match=fuzzy_match,
                )
                for query in queries
            ]
        else:  # parallel prompting
            sqlss, value_map = await self._generate_sql(
                query=queries,
                table=db.get_table(),
                template_key=TemplateKey.SQL_GENERATION_2,
                fuzzy_match=fuzzy_match,
            )

        # print(f"sqlss: {sqlss}")
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

            top_ans, pred_sqls = majority_vote(nsqls=sqls, pred_answer_list=preds)
            top_ans = self._process_list(top_ans, verbose) if isinstance(top_ans, list) else '[]'
            unit = self.parser.parse_sql_unit(pred_sqls[0][0])
            if verbose: print(f"A{idx+1}: {top_ans}. {unit}\n{'*'*75}")

            answers.append((top_ans, unit))

        return answers, value_map

    async def _evaluate_soundness(self, reasoning: str):
        evaluation = await self._call_api_2(
            prompt=[
                {
                    "role": "system",
                    "content": """You are an amazing logician. You are given a sequence of logical deduction based on real-world data. 
                You need to evaluate the soundness of the reasoning and fix the reasoning while still RETAIN the core idea and be as informative as possible in the following format.
                \{
                    explain: "<TODO: explain why the reasoning is sound or not sound>"   
                    revised: "<TODO: revised reasoning>"
                \}""",
                },
                {"role": "user", "content": reasoning},
            ],
            model=Model.GPT4,  # 4
        )

        return self.parser.parse_evaluation(evaluation[0])

    def _build_dec_prompt(
        self, sub_queries: list, answers: list, table: pd.DataFrame, question: str
    ):
        dec_prompt = self.prompter.build_prompt(
            template_key=TemplateKey.DEC_REASONING_2,
            table=table,
            question=question,
        )
        dec_prompt.extend(
            [
                {"role": "user", "content": "\n".join(sub_queries[:-1])},
                {"role": "assistant", "content": "\n".join(answers)},
                {"role": "user", "content": sub_queries[-1]},
            ]
        )
        return dec_prompt

    def _process_list(self, lst: list, verbose=False):
        try:
            if len(lst) > 30:
                # sometimes the answer is too long to fit into the prompt
                new_lst = [
                    x for x in lst if isinstance(x, (int, float)) and not math.isnan(x)
                ]
                if not new_lst and isinstance(lst[0], str):
                    return str(random.sample(lst, 10))  # array full of string
                new_lst = [round(x, 3) for x in new_lst]
                return f"Ranging from {str(min(new_lst))} to {str(max(new_lst))}, with average {str(sum(new_lst)/len(new_lst))}"
            return str([round(x, 3) if isinstance(x, float) else x for x in lst])
        except Exception as e:
            if verbose:
                print("error with list ans: ", e)
            return "[]"

    # @log_decorator
    async def reason(
        self,
        claim: str,
        table: pd.DataFrame,
        verbose=False,
        fuzzy_match=False,
        more_attrs: list = [],
    ):
        """
        Reasoning pipeline for CoT
        Input: claim, table
        Output: justification
        """
        db = NeuralDB(
            tables=[table], add_row_id=False, normalize=False, lower_case=False
        )
        # take first query from suggested queries
        suggestions, vis_tasks, _, attributes = await self._suggest_queries(
            claim=claim, table=db.get_table_df(), more_attrs=more_attrs
        )
        # update table with relevant attributes
        if attributes:
            db.update_table(attributes)
        if verbose:
            print(f"generated queries: {suggestions}")
            if attributes:
                print(f"mapped attributes: {attributes}")

        reason_map = []
        for idx, query in enumerate(suggestions[:1]):
            # decompose queries
            sub_queries = await self._decompose_query(query)
            if verbose:
                print(f"steps of reasoning: {sub_queries}")

            # execute sql corresponding to each subquery (up to the second last one)
            answers, value_map = await self._exec_sqls_from_sub_queries(
                db=db,
                queries=sub_queries[:-1],
                is_sequential=False,
                verbose=verbose,
                fuzzy_match=fuzzy_match,
            )
            sub_queries = [f"Q{i+1}: {query}" for i, query in enumerate(sub_queries)]
            answers = [
                f"A{i+1}: {ans}. {unit}" for i, (ans, unit) in enumerate(answers)
            ]
            # generate prompt for decomposed reasoning
            dec_prompt = self._build_dec_prompt(
                sub_queries, answers, db.get_table_df(), query
            )
            # if verbose: print(f"full prompt:\n{dec_prompt}")
            answers.extend(await self._call_api_2(dec_prompt))

            response = await self._call_api_2(
                prompt=[
                    {
                        "role": "system",
                        "content": """You are an amazing rhetorician. You are given a sequence of questions and answers that aims to tackle an ultimate question step by step. 
                                    You need to reframe the sequence to make it look like a coherent, smooth paragraph of logical deduction.""",
                    },
                    {
                        "role": "user",
                        "content": "\n".join(
                            query + "\n" + answer
                            for query, answer in zip(sub_queries, answers)
                        ),
                    },
                ],
                model=Model.GPT3,  # 4
            )
            justification = response[0]

            # use GPT4 to evaluate whether the reasoning is sound or not, then revise the reasoning if needed
            # justification = await self._evaluate_soundness(justification)
            reason_map.append(
                {
                    "query": query,
                    "visualization": vis_tasks[idx],
                    "reasoning_steps": sub_queries,
                    "justification": justification,
                    "value_map": value_map,
                }
            )

        if verbose: print(f"final justifications: {reason_map}\n{'@'*75}")
        
        return {
            "suggestions": reason_map,
            "sub_table": {
                "data": db.get_table_df()
            },
            "attributes": attributes
        }
    
    async def reason_2(
                self, claim_map: ClaimMap, 
                df: pd.DataFrame,
                verbose=False, 
                fuzzy_match=True,
            ):
        country_attr, date_attr = claim_map.mapping["country"], claim_map.mapping["date"]
        # claim_map already contains the elements constituting the queries
        queries, answers = [], []
        for country in claim_map.country:
            for datetime in claim_map.date:
                for category in claim_map.value:
                    if datetime.startswith('@('):
                        dates = list(claim_map.mapping[datetime])
                        date_mask = df[date_attr].isin(dates)
                        date_name = f"in {', '.join([str(date) for date in dates])}"
                    elif '-' in datetime:
                        start, end = datetime.split('-')
                        date_mask = (df[date_attr] >= int(start)) & (df[date_attr] <= int(end))
                        date_name = f"from {start} to {end}"
                    elif datetime[-1] == 's': # 1960s, 1980s
                        date_mask = (df[date_attr] >= int(datetime[:-1])) & (df[date_attr] <= int(datetime[:-1]) + 9)
                        date_name = f"in the {datetime}"
                    elif datetime.isdigit():
                        date_mask = df[date_attr] == int(datetime)
                        date_name = f"in the {datetime}"

                    category_name = claim_map.mapping[category] 

                    if country.startswith("@("):                        
                        country_name = claim_map.mapping[country]
                        country_mask = df[country_attr].isin(country_name)

                        if any(p in country for p in ['with', 'Countries of']):
                            query = f"What are the {country[2:-2]} {date_name}?"
                        else:
                            query = f"What is the {country[2:-2]} of {category_name} {date_name}?"
                        queries.append(f"Q{len(queries)+1}: {query}")
                        answers.append(f"A{len(answers)+1}: {country_name}")
                    else:
                        country_name = claim_map.mapping[country] or country
                        country_mask = df[country_attr] == country_name

                    val = df[date_mask & country_mask][category_name].values
                    val = self._process_list(val, verbose)
                    query = f"Q{len(queries)+1}: What is the {category_name} of {country_name} {date_name}?"
                    queries.append(query)
                    answer = f"A{len(answers)+1}: {str(val)}"
                    answers.append(answer)
        queries.append(f"Q{len(queries)+1}: {claim_map.rephrase}")
        # if verbose: print(f"queries: {queries}\nanswers: {answers}")
        dec_prompt = self._build_dec_prompt(queries, answers, df, claim_map.rephrase)
        # if verbose: print(f"full prompt:\n{dec_prompt}")
        answers.extend(await self._call_api_2(dec_prompt))

        msg = [
                {"role": "system", "content": """You are an amazing rhetorician and logician. You are given a sequence of questions and answers that aims to provide insight into the data. 
                First, reframe the sequence into a logical, coherent paragraph of deduction. Make adjustment to the logic if needed.
                Second, suggest some interesting patterns in the data within the deduction."""},
                {"role": "user", "content": "\n".join(query + "\n" + answer for query, answer in zip(queries, answers))},
            ]
        print(f"msg: {msg}")
        response = await self._call_api_2(
                                prompt = msg,
                                model=Model.GPT4 # 4
                            )
        justification = response[0]
        # justification = await self._evaluate_soundness(justification)
        if verbose: print(f"justification: {justification}")
        return justification
                        

async def main():
    data_matcher = DataMatcher(datasrc="../Datasets")
    table_reasoner = TableReasoner(datamatcher=data_matcher)
    query = "Albania had not had good economic prospect 20 years ago compared to East Asian Countries."
    await table_reasoner._suggest_queries_2(UserClaimBody(userClaim=query), verbose=True)
    # data = await table_reasoner._tag_claim(
    #     query,
    #     TemplateKey.CLAIM_TAGGING_2,
    #     model=Model.GPT_TAG_4,
    #     verbose=True,
    #     fewshot_samples=6,
    #     gen_samples=7,
    # )

    # tasks = [
    #         table_reasoner._tag_claim(
    #                 query, TemplateKey.CLAIM_TAGGING_2,
    #                 model=Model.GPT4, verbose=True, samples=5
    #             ),
    #         table_reasoner._tag_claim(
    #                 query, TemplateKey.CLAIM_TAGGING_2,
    #                 model=Model.GPT_TAG_3, verbose=True, samples=10
    #             ),
    #         table_reasoner._tag_claim(
    #             query, TemplateKey.CLAIM_TAGGING_2,
    #             model=Model.GPT_TAG_4, verbose=True, samples=10
    #         )]
    # await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
