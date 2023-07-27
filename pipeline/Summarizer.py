import openai
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from Credentials.info import *
import pandas as pd
import os
import json

# offline handle datasets

# # set credentials
openai.organization = organization
openai.api_key = openai_api
database_path = "../Datasets/"

def prompt_description(file_name: str):
    file_path = database_path + file_name
    df = pd.read_csv(file_path)
    
    # prepare metadata of the file
    meta_data = get_meta_data(df, file_name)
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

def get_meta_data(df, file_name):
    # this function is to be changed
    max_number_of_attributes = 12 # free to set
    return {
        'name': file_name,
        'attributes': df.columns.tolist()[:max_number_of_attributes]
    }

def summarize_all():
    descriptions = []
    for file_name in os.listdir(database_path):
        descriptions.append(summarize(file_name))
    return descriptions

def summarize(file_name: str):
    return {
        "name": file_name,
        "description": prompt_description(file_name)
    }
    
file_name = "colleges.csv"
# Writing to sample.json
with open("../DataDescription/description.json", "r+") as outfile:
    # load and write new data
    file_data = json.load(outfile)
    file_data.append(summarize(file_name))

    # Sets file's current position at offset.
    outfile.seek(0)
    # convert back to json.
    json.dump(file_data, outfile, indent = 4)