import openai
from debater_python_api.api.debater_api import DebaterApi
from Credentials.info import *
import requests
import json

# set credentials
openai.organization = organization
openai.api_key = openai_api
debater_api = DebaterApi(Debater_api)
claim_boundaries_client = debater_api.get_claim_boundaries_client()

# sentence to extract claims
sentence = "The US's 50 million citizens have been placing a burden tax of 100M USD on the government every year."

""" 
    Check if there is check-worthy claims in the sentence
    using ClaimBuster API
"""
# Define the endpoint (url) with the claim formatted as part of it, api-key (api-key is sent as an extra header)
api_endpoint = f"https://idir.uta.edu/claimbuster/api/v2/score/text/{sentence}"
request_headers = {"x-api-key": Claim_Buster_api}

# Send the GET request to the API and store the api response
api_response = requests.get(url=api_endpoint, headers=request_headers).json()

# if the score is > .5 --> checkworthy
if api_response['results'][0]['score'] > .5:
    """
        create prompt chat to detect if the claim is "statistically interesting".
        Somehow ChatGPT does well with this abstract definition of statistical interest.
        It defines the term as "a claim or statement that raises curiosity or suggests 
        the need for statistical analysis to investigate its validity".
    """
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": """Label the following claims as 'Y' if they are statistically interesting, otherwise 'N'. Give detailed explanation."""},
            {"role": "user", "content": f"{sentence}"},
        ]
    )
    print(res['choices'][0]['message']['content'])
    # if res['choices'][0]['message']['content'] == 'Y':
    #     print(f"statistically interesting: {api_response['results'][0]['score']}")

    #     # extract the boundary
    #     sentences = [sentence]
    #     boundaries_dicts = claim_boundaries_client.run(sentences)
    #     for i in range(len(sentences)):
    #         print ('In sentence: '+sentences[i])
    #         print ('['+str(boundaries_dicts[i]['span'][0])+', '+str(boundaries_dicts[i]['span'][1])+']: '
    #             +boundaries_dicts[i]['claim'])
    #         print ()
    # else:
    #     print("statistically uninteresting")
    
else:
    print(f"non-checkworthy: {api_response['results'][0]['score']}")