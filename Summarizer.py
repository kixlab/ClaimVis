import openai
from Credentials.info import *
import pandas as pd
import os
import json

# offline handle datasets

# # set credentials
openai.organization = organization
openai.api_key = openai_api
database_path = "./Datasets/"

def prompt_description(file_name: str):
    file_path = database_path + file_name
    df = pd.read_csv(file_path)
    
    # prepare metadata of the file
    max_number_of_attributes = 12
    meta_data = {
        'name': file_name,
        'attributes': df.columns.tolist()[:max_number_of_attributes]
    }
    # Prepare the dataset for summarization
    message = f"Infer what the purpose of the following dataset given its metada:\n\n{str(meta_data)}"
    # Maximum length of the generated summary
    max_summary_length = 150

    # Call GPT-3.5 to generate the summary
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the appropriate engine name
        messages=[
            {"role":"user", "content": message}
        ],
        max_tokens=max_summary_length
    )

    return response['choices'][0]['message']['content']

descriptions = []
for file_name in os.listdir(database_path):
    # write res into a json file
    dictionary = {
        "name": file_name,
        "description": prompt_description(file_name)
    }

    descriptions.append(dictionary)

# Serializing json
json_object = json.dumps(descriptions, indent=4)
    
# Writing to sample.json
with open("./DataDescription/description.json", "w") as outfile:
    outfile.write(json_object)