import openai
from debater_python_api.api.debater_api import DebaterApi
from Credentials.info import *
import requests
import json
import re

# set credentials
openai.organization = organization
openai.api_key = openai_api
debater_api = DebaterApi(Debater_api)
claim_boundaries_client = debater_api.get_claim_boundaries_client()

class ClaimDetector():
    def __init__(self):
        pass
    
    def detect(self, sentence: str, verbose: bool=False):
        """ 
            Check if there is check-worthy claims in the sentence
            using ClaimBuster API
        """
        # Define the endpoint (url) with the claim formatted as part of it, api-key (api-key is sent as an extra header)
        api_endpoint = f"https://idir.uta.edu/claimbuster/api/v2/score/text/{sentence}"
        request_headers = {"x-api-key": Claim_Buster_api}

        # Send the GET request to the API and store the api response
        api_response = requests.get(url=api_endpoint, headers=request_headers).json()
        if verbose: print(api_response)

        # if the score is > .5 --> checkworthy
        if api_response['results'][0]['score'] > .5:
            """
                create prompt chat to detect if the claim is check-worthy and data related.
            """
            res = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """Label the following claims as 'Y' if they are data related, otherwise 'N' according to the following format.
                    \{
                        "verdict": "<TODO: Y or N>",
                        "explain": "<TODO: explain why the claim can be verified using data or not>"
                    \}"""},
                    {"role": "user", "content": f"{sentence}"},
                ]
            )["choices"][0]['message']['content']

            verdict = re.search(r'"verdict": "(Y|N)"', res).group(1)
            if verdict == 'Y':
                if verbose: 
                    print(f"statistically interesting: {api_response['results'][0]['score']}")
                    explain = re.search(r'"explain": "(.+)"', res).group(1)
                    print(f"explain: {explain}")

                # extract the boundary
                sentences = [sentence]
                boundaries_dicts = claim_boundaries_client.run(sentences)
                if verbose:
                    print ('In sentence: '+sentences[0])
                    print ('['+str(boundaries_dicts[0]['span'][0])+', '+str(boundaries_dicts[0]['span'][1])+']: '
                        +boundaries_dicts[0]['claim'])
                    print ()
                return boundaries_dicts[0]['claim'], api_response['results'][0]['score']
            else:
                if verbose: print("statistically unrelated")
                # negative means unrelated to data
                return "", -api_response['results'][0]['score']
            
        else:
            if verbose: print(f"non-checkworthy: {api_response['results'][0]['score']}")
            return "", api_response['results'][0]['score']

if __name__ == "__main__":
    detector = ClaimDetector()
    sentence = "The average cost of a college education is $20,000 per year."
    detector.detect(sentence)