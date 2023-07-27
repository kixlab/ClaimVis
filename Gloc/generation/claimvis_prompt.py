"""ClaimVis Prompts."""

import enum
import random
from typing import Dict, Tuple
import pandas as pd
import copy
from utils.table import table_linearization,twoD_list_transpose
from utils.json import NoIndent, MyEncoder
import json
import os

class TemplateKey(enum.Enum):
    QA = 'qa'
    POT = 'pot'
    STATISTA_SUMMARY = 'statista_summary'
    PEW_SUMMARY = 'pew_summary'

class PromptBuilder(object):
    def __init__(self) -> None:
        # respectively read pre-prompt files in fewshots folders and 
        # set corresponding attributes
        _path_ = "generation/fewshots/"
        for file_name in os.listdir(_path_):
            attr_name = '_' + file_name.upper()[:-5] + '_'
            with open(_path_ + file_name, "r") as file:
                setattr(self, attr_name, json.loads(file.read()))
        

    def _get_template(self, template_key):
        """Returns a template given the key which identifies it."""
        pass
        

    def build_prompt(
            self, 
            template_key,
            table: pd.DataFrame,
            question: str = None,
            title: str = None,
            num_rows: int = 3,
            **kwargs
        ):
        """Builds a prompt given a table, question and a template identifier."""
        template = self._get_template(template_key)