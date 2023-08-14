import openai
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from Credentials.info import *
import pandas as pd
import os
import json

# # set credentials
openai.organization = organization
openai.api_key = openai_api

# offline handle datasets
class Summarizer:
    def __init__(self, datasrc: str):
        self.datasrc = datasrc

    def prompt_description(self, file_name: str):
        file_path = os.path.join(self.datasrc, file_name)
        df = pd.read_csv(file_path)
        
        # prepare metadata of the file
        meta_data = self.get_meta_data(df, file_name)
        message = f"Given the meta data of a dataset, describe what the dataset is about, including its data attributes' datat type and description:\n\n{str(meta_data)}"
        # Maximum length of the generated summary
        max_summary_length = 1000

        # Call GPT-3.5 to generate the summary
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the appropriate engine name
            messages=[
                {"role":"user", "content": message}
            ],
            max_tokens=max_summary_length
        )

        return response['choices'][0]['message']['content']

    def get_meta_data(self, df, file_name, max_number_of_attributes=100):
        # this function is to be changed
        return {
            'name': file_name,
            'attributes': df.columns.tolist()[:max_number_of_attributes]
        }

    def summarize_all(self):
        descriptions = []
        for file_name in os.listdir(self.datasrc):
            descriptions.append(self.summarize(file_name))
        return descriptions

    def summarize(self, file_name: str):
        return {
            "name": file_name,
            "description": self.prompt_description(file_name)
        }
    
if __name__ == "__main__":
    # Write data_dict to ../Datasets/description/desc.json
    with open("../Datasets/description/desc.json", "r+") as outfile:
        data_list = json.load(outfile)
        data_dict = {item['name']: item for item in data_list}
        outfile.seek(0)
        json.dump(data_dict, outfile, indent=4)
    