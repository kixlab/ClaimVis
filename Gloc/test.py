import os
import sys
sys.path.append("..")

# from generation.dater_prompt import PromptBuilder
from generation.claimvis_prompt import *
# from generation.dater_generator import Generator
from generation.deplot_prompt import build_prompt
from utils.table import table_linearization
from processor.ans_parser import *
from utils.llm import *
import pandas as pd
import json
import csv
from common.functionlog import log_decorator

# file_name = "owid-energy-data.csv"
file_name = "movies-w-year.csv"
dataset_path = "../Datasets/" + file_name

@log_decorator
def test_select_x_col_prompt():
    prompter = PromptBuilder(1000)
    df = pd.read_csv(dataset_path)

    generate_prompt = prompter.build_generate_prompt(
        table = df,
        question = "The electricity consumption of US is greater than that of China this year",
        title = file_name,
        num_rows = 50,
        select_type = 'col'
    )
    print(generate_prompt)

# test_select_x_col_prompt()

@log_decorator
def test_build_prompt(template_key, table, question):
    table = """Year | Democrats | Republicans | Independents
    2004 | 68.1% | 45.0% | 53.0%"""
    question = "In which year republicans have the lowest favor rate?"

    prompt = build_prompt(template_key, table, question)
    print(prompt)

_TABLE = """Year | Democrats | Republicans | Independents
2004 | 68.1% | 45.0% | 53.0%
2006 | 58.0% | 42.0% | 53.0%
2007 | 59.0% | 38.0% | 45.0%
2009 | 72.0% | 49.0% | 60.0%
2011 | 71.0% | 51.2% | 58.0%
2012 | 70.0% | 48.0% | 53.0%
2013 | 72.0% | 41.0% | 60.0%"""
# test_build_prompt(TemplateKey.QA, _TABLE, question="What's the highest man?")

@log_decorator
def test_table_linearization():
    # Create a sample DataFrame
    df = pd.read_csv(dataset_path)

    # Test 'tapex' format
    tab = table_linearization(df, 'tapex')
    print(tab)

# test_table_linearization()

@log_decorator
def test_prompt_builder_1():
    prmpt = PromptBuilder()
    print(prmpt._DEC_REASONING_)

# test_prompt_builder()

@log_decorator
def test_prompt_builder_2(
        question = "How many movies have funny adjective in their names",
        template_key = TemplateKey.COL_DECOMPOSE
    ):
    # set prompt builder
    prompter = Prompter()
    # load dataset
    df = pd.read_csv(dataset_path)
    # return prompt
    prompt = prompter.build_prompt(
        template_key=template_key,
        table=df.iloc[:5, :5],
        question=question,
        title=None
    )
    
    # print(prompt)
    return prompt

# test_prompt_builder_2()

@log_decorator
def test_call_api_1(
        question = "There are two movies that have higher gross than The Phantom.",
        template_key = TemplateKey.COL_DECOMPOSE
    ):
    prompt = test_prompt_builder_2(
                question=question,
                template_key=template_key
            )
    response = call_model(
        model=Model.GPT3,
        use_code=False,
        temperature=0,
        max_decode_steps=500,
        prompt=prompt,
        samples=1
    )
    print(question, response)
    # print(prompt)
    return response

# test_call_api_1(template_key=TemplateKey.ROW_DECOMPOSE)

@log_decorator
def test_call_api_2(prompt):
    response = call_model(
        model=Model.GPT3,
        use_code=False,
        temperature=0,
        max_decode_steps=500,
        prompt=prompt,
        samples=1
    )
    # print(question, response)
    return response

@log_decorator
def test_dec_reasoning(
        # question = "Is the largest gross twice the amount of the lowest gross?"
        question = "Are there only two movies that have greater grosses than 300000000?"
    ):
    # Create a sample DataFrame
    df = pd.read_csv(dataset_path)

    # decompose query into subqueries
    decomposed_ans = test_call_api_1(
                    question=question,
                    template_key=TemplateKey.QUERY_DECOMPOSE
                )
    parser, prompter = AnsParser(), Prompter()
    sub_queries = parser.parse_dec_reasoning(decomposed_ans[0])

    # inject cot prompt with sub queries
    dec_prompt = prompter.build_prompt(
                    template_key=TemplateKey.DEC_REASONING,
                    table=df,
                    question=question,
                    num_rows=3
                )
    for query in sub_queries:
        dec_prompt.append({
            "role": "user", 
            "content": query
        })
        response = test_call_api_2(dec_prompt)
        dec_prompt.append({
            "role": "assistant",
            "content": response[0]
        })
    print(dec_prompt)
    return dec_prompt[-1]

# test_dec_reasoning()

test_call_api_1(
    question=  "The US has the highest electricity consumption in 2010",
    template_key=TemplateKey.QUERY_GENERATION
)
