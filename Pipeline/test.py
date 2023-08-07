import os
import sys
sys.path.append("../Gloc")
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
from processor.table_truncate import RowDeleteTruncate
from processor.table_linearize import IndexedRowTableLinearize

# file_name = "owid-energy-data.csv"
file_name = "movies-w-year.csv"
dataset_path = "../Datasets/" + file_name
df = pd.read_csv(dataset_path)
table = df.iloc[:10, :10]

# save df[:3, :] to Datasets/owid-energy-data-2.csv
# df.iloc[:3, :].to_csv("../Datasets/owid-energy-data-2.csv", index=False)


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
        table: pd.DataFrame = None,
        question = "How many movies have funny adjective in their names",
        template_key = TemplateKey.COL_DECOMPOSE
    ):
    # set prompt builder
    prompter = Prompter()

    # return prompt
    prompt = prompter.build_prompt(
        template_key=template_key,
        table=table,
        question=question,
        title=None
    )
    
    # print(prompt)
    return prompt

# test_prompt_builder_2()

@log_decorator
def test_call_api_1(
        table: pd.DataFrame = None,
        question = "The second movie has an IMDB rating higher than the third movie.",
        template_key = TemplateKey.COL_DECOMPOSE
    ):
    prompt = test_prompt_builder_2(
                table=table,
                question=question,
                template_key=template_key
            )
    print(prompt)

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

test_call_api_1(template_key=TemplateKey.NSQL_GENERATION, table=table)

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
        question = "Are there movies with much higher gross than the Phantom?"
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

# test_call_api_1(
#     question=  "No movie has a rating better than 66.6.",
#     template_key=TemplateKey.QUERY_GENERATION_2
# )


@log_decorator
def test_break_big_columns_1(
    question = "The US's economy is larger than China's.",
):
    # Create a sample DataFrame
    df = pd.read_csv(dataset_path)
    df_dict = {
        'header': df.columns.tolist()[:20],
        'rows': [row[:20] for row in df.values.tolist()]
    }

    # truncate table rows
    truncator = RowDeleteTruncate(
        table_linearize=IndexedRowTableLinearize(), 
        # max_input_length=1024
    )
    truncator.truncate_table(
        table_content=df_dict,
        question=question,
        answer=[]
    )

    new_table = pd.DataFrame(columns=df_dict['header'], data=df_dict['rows'])

    # use LLM to infer related columns
    decomposed_cols = test_call_api_1(
        table=new_table,
        question=question,
        template_key=TemplateKey.COL_DECOMPOSE
    )

    print(decomposed_cols)

# test_break_big_columns_1()

@log_decorator
def test_pipeline_1(
    question = "The US's economy is larger than China's."
):
    # without the generation module
    # Create a sample DataFrame
    df = pd.read_csv(dataset_path)
    


    