"""ClaimVis Prompts."""
import enum
import random
from typing import Dict, Tuple
import pandas as pd
import copy
from utils.table import table_linearization, twoD_list_transpose 
from utils.json import NoIndent, MyEncoder
import json
import os
import math
from generation.dater_prompt import PromptBuilder
from generation.binder_prompt import PromptBuilder as BinderPromptBuilder

class TemplateKey(str, enum.Enum):
    SUMMARY = "sum"

    ROW_DECOMPOSE = 'row'
    COL_DECOMPOSE = 'col'

    QUERY_DECOMPOSE = 'que'

    COT_REASONING = 'cot'
    DEC_REASONING = 'dec'
    DEC_REASONING_2 = 'dec2'

    QUERY_GENERATION = 'gen'
    QUERY_GENERATION_2 = 'gen2'
    QUERY_GENERATION_3 = 'gen3'

    NSQL_GENERATION = 'nsql'
    SQL_GENERATION = 'sql'
    SQL_GENERATION_2 = 'sql2'

    CLAIM_EXTRACTION = 'ext'
    CLAIM_TAGGING = 'tag'
    CLAIM_TAGGING_2 = 'tag2'

class Prompter(object):
    def __init__(self) -> None:
        # respectively read pre-prompt files in fewshots folders and 
        # set corresponding attributes
        self.attributes = []

        _path_ = "../Gloc/generation/fewshots/"
        for file_name in os.listdir(_path_):
            attr_name = '_' + file_name.upper()[:-5] + '_'
            self.attributes.append(attr_name)

            with open(_path_ + file_name, "r") as file:
                setattr(self, attr_name, json.loads(file.read()))        

    def _get_template(self, template_key):
        """Returns a template given the key which identifies it."""
        for attr in self.attributes:
            if getattr(TemplateKey, attr[1:-1]) == template_key:
                return getattr(self, attr)

    def build_prompt(
            self, 
            template_key,
            table: pd.DataFrame,
            question: str or list[str] = None,
            title: str = None,
            num_rows: int = 3,
            **kwargs
        ):
        """
            Builds a prompt given a table, question and a template identifier.
            This is a wrapper for dater prompt builder's old functions and 
            some new modules 
        """
        pb = PromptBuilder() # dater promptbuilder
        bd = BinderPromptBuilder(None) # binder promptbuilder

        template = self._get_template(template_key)
        if template_key in [TemplateKey.COL_DECOMPOSE, TemplateKey.ROW_DECOMPOSE]:
            template.append({
                "role": "user", 
                "content": pb.build_generate_prompt(
                    table=table,
                    question=question,
                    title=title,
                    num_rows=num_rows,
                    select_type=template_key
                )
            })
        elif template_key in [TemplateKey.QUERY_DECOMPOSE, TemplateKey.QUERY_GENERATION]:
            template.append({
                "role": "user",
                "content": question
            })
        elif template_key in [TemplateKey.DEC_REASONING, TemplateKey.DEC_REASONING_2]:
            template.append({
                "role": "system",
                "content": pb._select_x_wtq_end2end_prompt(
                        question=question,
                        caption=title,
                        df=table,
                        num_rows=num_rows
                    )
            })
        elif template_key == TemplateKey.QUERY_GENERATION_2:
            template.append({
                "role": "user",
                "content": f"statement: {question}"
            })
        elif template_key in [TemplateKey.NSQL_GENERATION, TemplateKey.SQL_GENERATION, TemplateKey.SQL_GENERATION_2]:
            template.append({
                "role": "user",
                "content": bd.build_one_shot_prompt(
                    prompt_type=("question" if isinstance(question, str) else "questions", ),
                    table=table,
                    question=question,
                    answer_text=None,
                    nsql=""
                )
            })
        elif template_key == TemplateKey.QUERY_GENERATION_3:
            template.append({
                "role": "user",
                "content": bd.build_one_shot_prompt(
                    prompt_type=("statement",),
                    table=table,
                    question=question,
                    answer_text=None,
                    nsql=""
                )
            })
        elif template_key == TemplateKey.CLAIM_EXTRACTION:
            template.append({
                "role": "user",
                "content": f"""PARAGRAPH: {kwargs['paragraph']}\n   SENTENCE: {kwargs['userClaim']}"""
            })
        elif template_key in [TemplateKey.CLAIM_TAGGING, TemplateKey.CLAIM_TAGGING_2]:
            template.append({
                "role": "user",
                "content": f"""SENTENCE: {question}"""
            })
        else:
            template = []

        return template
               
        