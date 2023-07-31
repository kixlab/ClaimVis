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

class TemplateKey(str, enum.Enum):
    SUMMARY = "sum"
    ROW_DECOMPOSE = 'row'
    COL_DECOMPOSE = 'col'
    QUERY_DECOMPOSE = 'que'
    COT_REASONING = 'cot'
    DEC_REASONING = 'dec'
    QUERY_GENERATION = 'gen'
    

class Prompter(object):
    def __init__(self) -> None:
        # respectively read pre-prompt files in fewshots folders and 
        # set corresponding attributes
        self.attributes = []

        _path_ = "generation/fewshots/"
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
            question: str = None,
            title: str = None,
            num_rows: int = 10,
            **kwargs
        ):
        """
            Builds a prompt given a table, question and a template identifier.
            This is a wrapper for dater prompt builder's old functions and 
            some new modules 
        """
        pb = PromptBuilder() # dater promptbuilder

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
        elif template_key == TemplateKey.DEC_REASONING:
            template.append({
                "role": "system",
                "content": pb._select_x_wtq_end2end_prompt(
                        question=question,
                        caption=title,
                        df=table,
                        num_rows=num_rows
                    )
            })

        return template
               