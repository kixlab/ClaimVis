import os
# from generation.dater_prompt import PromptBuilder
from generation.claimvis_prompt import PromptBuilder
from generation.dater_generator import Generator
from generation.deplot_prompt import *
from utils.table import table_linearization
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
        question = "The electricity consumption of US is greater than that of China this year.",
        title = file_name,
        num_rows = 50,
        select_type = 'all'
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

test_prompt_builder()