import openai
from debater_python_api.api.debater_api import DebaterApi
from Credentials.info import *
import requests
import json

# set credentials
openai.organization = organization
openai.api_key = openai_api
res = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": """Explain TF-IDF to me."""},
    ]
)
print(res['choices'][0]['message']['content'])
