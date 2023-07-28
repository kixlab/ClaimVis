import os
import sys
sys.path.append("..")

# from generation.dater_prompt import PromptBuilder
from generation.claimvis_prompt import *
# from generation.dater_generator import Generator
from generation.deplot_prompt import build_prompt
from utils.table import table_linearization
from utils.llm import *
import pandas as pd
import json
import csv

# file_name = "owid-energy-data.csv"
file_name = "movies-w-year.csv"
dataset_path = "../Datasets/" + file_name

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

def test_table_linearization():
    # Create a sample DataFrame
    df = pd.read_csv(dataset_path)

    # Test 'tapex' format
    tab = table_linearization(df, 'tapex')
    print(tab)

# test_table_linearization()

def test_prompt_builder():
    prmpt = PromptBuilder()
    print(prmpt._COT_DEC_REASONING_)

# test_prompt_builder()

def test_col_decompose(question = "How many movies have funny adjective in their names"):
    # set template key for the correct choice of prompt
    key = TemplateKey.QUERY_DECOMPOSE
    # set prompt builder
    prompter = Prompter()
    # load dataset
    df = pd.read_csv(dataset_path)
    # return prompt
    prompt = prompter.build_prompt(
        template_key=key,
        table=df.iloc[:10, :10],
        question=question,
        title=None
    )
    
    print(prompt)
    return prompt

# test_col_decompose()

def test_call_api(question = "Is the US' electricity consumption is larger than that from China?"):
    prompt = test_col_decompose(question=question)
    response = call_model(
        model=Model.GPT3,
        use_code=False,
        temperature=0,
        max_decode_steps=500,
        prompt=prompt,
        samples=1
    )
    print(question, response)

test_call_api()
