import sys
sys.path.append("../Gloc")
sys.path.append("..")

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Gloc.utils.async_llm import *
from Summarizer import Summarizer
import numpy as np
import pandas as pd
import json
from pathlib import Path
import os

class DataMatcher(object):
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    def __init__(self, datasrc: str=None, summarize: bool=True):
        self.summarizer = Summarizer(datasrc=datasrc)
        
        if not datasrc:
            return 
        # if datasrc is specified, load the datasets
        self.datasrc = datasrc
        self.datasets = [filename for filename in os.listdir(self.datasrc) if filename.endswith('.csv')]

        with open(f'{self.datasrc}/description/desc.json', 'r+') as openfile:
            self.description = json.load(openfile)

            # Get the list of files in the directory once
            description_dir = Path(self.datasrc) / 'description'
            existing_files = set(os.listdir(description_dir))
            # prepare dataset embeddings and descriptions
            writeflag, self.dataset_embeddings, self.attrs_embeddings = False, [], dict()
            for dataset in self.datasets:
                if dataset not in self.description:
                    if summarize:
                        self.description[dataset] = self.summarizer.summarize(dataset)
                        writeflag = True
                                        
                else:
                    if "columns" not in self.description[dataset]:
                        self.description[dataset]["columns"] = pd.read_csv(f"{self.datasrc}/{dataset}").columns.tolist()
                        writeflag = True
                
                    if "embedding" not in self.description[dataset]:
                        self.description[dataset]["embedding"] = self.encode(self.description[dataset]['description']).tolist()
                        writeflag = True
                    else:
                        self.dataset_embeddings.append(self.description[dataset]["embedding"])

                embed_name = f"{dataset[:-5]}_column_embeddings.json"
                if embed_name not in existing_files:
                    column_embeddings = {col_name: self.encode(col_name).tolist() for col_name in self.description[dataset]["columns"]}
                    with open(description_dir / embed_name, 'w') as f:
                        json.dump(column_embeddings, f)    
                    self.attrs_embeddings[dataset] = column_embeddings     
                else:
                    with open(description_dir / embed_name, 'r') as f:
                        self.attrs_embeddings[dataset] = json.load(f)

            # write summaries back to file
            if writeflag:
                openfile.seek(0)
                openfile.truncate(0)
                json.dump(self.description, openfile)
            openfile.close()

    async def find_top_k_datasets(
            self, claim: str, 
            k: int = 2, 
            method: str = "attr", 
            verbose: bool = True,
            keywords: list = None
        ):
        assert self.datasrc, "Datasrc not specified."

        relevant_attrs = []
        # Compute embeddings for the text and dataset descriptions
        if method == "cosine":
            similarities = self.similarity_batch(claim, self.dataset_embeddings)
        elif method == "idf":
            similarities = [self.idf_score(claim, self.description[dataset]['description']) for dataset in self.datasets]
        elif method == "attr": # the most accurate match
            embed = self.encode(claim)
            score_batches = [self.similarity_batch(embed, list(self.attrs_embeddings[dataset].values()))\
                                                             for dataset in self.datasets]
            similarities = [max(batch) for batch in score_batches]
        elif method == "gpt": # semantic relatedness 
            if not keywords:
                prompt = [
                    {"role": "system", "content": "You are an amazing extractor. Given a claim, your task is to extract keywords from it and generate more relevant keywords. Also please rate the relevance of each keyword to the claim on a scale of 0 to 1, with all scores summed to 1. Think step by step in the 'Reason' field."},

                    {"role": "user", "content": "The US economy is larger than China's."},
                    {"role": "assistant", "content": """{
                        "Reason": " The claim is about the economy of the US and China. Some relevant keywords are 'economy', 'US', 'China'. 'economy' is related to 'GDP', 'Trade Volume', 'GDP per capita'. 'US' and 'China' are related to 'country'.",
                        "Keywords": ["economy", "GDP", "Trade Volume", "GDP per capita", "US", "China", "country"],
                        "Scores": [0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1]
                    }"""},

                    {"role": "user", "content": "After hitting an all time high of 445$ in July 2021, TSLA price drops to 355$ and stays there for a few month."},
                    {"role": "assistant", "content": """{
                        "Reason": "The claim is about the price of TSLA. Some relevant keywords are 'price', 'TSLA', 'July', '2021'. 'TSLA price' is related to 'stock price', 'price change', 'stock market'. 'TSLA' is related to 'company', 'stock'. '2021' and 'July' are related to 'year', 'time', 'month'.",
                        "Keywords": ["price", "TSLA", "July", "2021", "stock price", "price change", "stock market", "company", "stock", "year", "time", "month"],
                        "Scores": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05]
                    }"""},

                    {"role": "user", "content": claim},
                ]
                response = await call_model(
                                model=Model.GPT3,
                                prompt=prompt,
                                temperature=0.0,
                                max_decode_steps=500,
                                samples=1
                            )
                answer = json.loads(response[0])
                answer["Keywords"].append(claim) # add claim into keywords also
                WEIGHT = 1/2
                seed_embeds, weights = self.encode(answer["Keywords"]), [*[score*(1-WEIGHT) for score in answer["Scores"]], WEIGHT]
            else:
                seed_embeds, weights = self.encode(keywords + [claim]), self.similarity_batch(claim, keywords + [claim])
                weights = [score/sum(weights) for score in weights]

            score_batches = [cosine_similarity(seed_embeds, list(self.attrs_embeddings[dataset].values())) \
                                                                for dataset in self.datasets]
            similarities = [np.average(np.max(batch, axis=1), weights=weights) \
                                                                for batch in score_batches]
        
        # Combine the dataset names, descriptions, similarities, and their corresponding index
        result = [[self.description[name]['name'], self.description[name]['description'], similarity, index] \
                            for index, (name, similarity) in enumerate(zip(self.datasets, similarities))]
        # Sort the result based on similarity in descending order
        result.sort(key=lambda x: x[2], reverse=True)

        top_k_datasets = result[:k]
        if method == "attr":
            for dataset_map in top_k_datasets:
                ind, name = dataset_map[3], dataset_map[0]
                dataset_map[3] = [attr for attr, score in zip(self.description[name]['columns'], score_batches[ind]) if score > top_k_datasets[0][2] * .8]
        elif method == "gpt":
            for dataset_map in top_k_datasets:
                ind, name = dataset_map[3], dataset_map[0]
                attr_scores = np.max(score_batches[ind], axis=0)
                max_score = max(attr_scores)
                dataset_map[3] = [attr for attr, score in zip(self.description[name]['columns'], attr_scores) if score > max_score * .9]
        else:
            # reset dataset_map[3] to empty list
            for dataset_map in top_k_datasets:
                dataset_map[3] = []

        if verbose: 
            print(f"Most relevant datasets using {method}")
            for dataset_name, _, similarity, relevant_attrs in top_k_datasets:
                print(f"Dataset: {dataset_name}, Similarity: {similarity:.2f}, Relevant Attributes: {relevant_attrs}")

        return top_k_datasets

    def similarity_score(self, phrase1, phrase2):
        phrase1_embedding = self.encode(phrase1) if isinstance(phrase1, str) else phrase1
        phrase2_embedding = self.encode(phrase2) if isinstance(phrase2, str) else phrase2
        similarity = cosine_similarity([phrase1_embedding], [phrase2_embedding])[0][0]
        return similarity
    
    def similarity_batch(self, phrase, batch_of_phrases):
        phrase_embedding = self.encode(phrase) if isinstance(phrase, str) else phrase
        batch_of_embeddings = self.encode(batch_of_phrases) \
                                    if isinstance(batch_of_phrases[0], str) else batch_of_phrases
        similarities = cosine_similarity([phrase_embedding], batch_of_embeddings)[0]
        return similarities

    def batch2batch(self, batch1, batch2):
        batch1_embeddings = self.encode(batch1) if isinstance(batch1[0], str) else batch1
        batch2_embeddings = self.encode(batch2) if isinstance(batch2[0], str) else batch2
        similarities = cosine_similarity(batch1_embeddings, batch2_embeddings)
        return similarities

    def idf_score(self, phrase1: str, phrase2: str):
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create the Document Term Matrix
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform([phrase1, phrase2])

        # Compute the cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    
    def attr_score_batch(self, phrase: any, attributes: list):
        return max(self.similarity_batch(phrase, attributes))
    
    def encode(self, phrase):
        if isinstance(phrase, str):
            return self.embedder.encode([phrase])[0]
        elif isinstance(phrase, list):
            return self.embedder.encode(phrase)
        return phrase
    
    def recompute_col_embeddings(self):
        for dataset in self.datasets:
            embed = {col_name: self.encode(col_name).tolist() for col_name in self.description[dataset]["columns"]}
            with open(f"{self.datasrc}/description/{dataset[:-5]}_column_embeddings.json", 'w') as f:
                json.dump(embed, f)
    
    def load_table(self, dataset: str, attributes: list=None, return_dict: bool=False):
        table = pd.read_csv(f"{self.datasrc}/{dataset}")
        if attributes:
            table = table[attributes]
        table.name = dataset
        if not return_dict:
            return table
        return {
            "name": dataset,
            "data": table
        }

async def main():
    matcher = DataMatcher(datasrc="../Datasets")
    input = "the percentage of sub-Saharan Africans living below the World Bank’s global poverty threshold of $1.90 per day dropped from 56% in 1990 to 40% in 2018."
    keywords = ["sub-Saharan Africans", "World Bank", "global poverty threshold", "1990", "2018"]
    # await matcher.find_top_k_datasets(
    #     claim=input,
    #     k=10,
    #     method="gpt",
    #     verbose=True,
    #     keywords=keywords
    # )
    matcher.recompute_col_embeddings()

if __name__ == "__main__":
    asyncio.run(main())