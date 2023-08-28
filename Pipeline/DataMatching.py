from functools import cache, lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Summarizer import Summarizer
from rapidfuzz import fuzz
import numpy as np
import pandas as pd
import json
import os

class DataMatcher(object):
    def __init__(self, datasrc: str=None, summarize: bool=False):
        self.summarizer = Summarizer(datasrc=datasrc)
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # prepare sentence embedder
        
        if not datasrc:
            return 
        # if datasrc is specified, load the datasets
        self.datasrc = datasrc
        self.datasets = [filename for filename in os.listdir(self.datasrc) if filename.endswith('.csv')]

        with open(f'{self.datasrc}/description/desc.json', 'r+') as openfile:
            self.description = json.load(openfile)

            # prepare dataset embeddings and descriptions
            writeflag, self.dataset_embeddings, self.attrs_embeddings = False, [], []
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
                if embed_name not in os.listdir(f"{self.datasrc}/description/"):
                    column_embeddings = [self.encode(col_name).tolist() for col_name in self.description[dataset]["columns"]]
                    with open(f"{self.datasrc}/description/{embed_name}", 'w') as f:
                        json.dump(column_embeddings, f)    
                    self.attrs_embeddings.append(column_embeddings)                    
                else:
                    with open(f"{self.datasrc}/description/{embed_name}", 'r') as f:
                        self.attrs_embeddings.append(json.load(f))

            # write summaries back to file
            if writeflag:
                openfile.seek(0)
                openfile.truncate(0)
                json.dump(self.description, openfile)
            openfile.close()

    def find_top_k_datasets(self, claim: str, k: int = 2, method: str = "attr", verbose: bool = True):
        assert self.datasrc, "Datasrc not specified."

        relevant_attrs = []
        # Compute embeddings for the text and dataset descriptions
        if method == "cosine":
            similarities = self.similarity_batch(claim, self.dataset_embeddings)
        elif method == "idf":
            similarities = [self.idf_score(claim, self.description[dataset]['description']) for dataset in self.datasets]
        elif method == "attr": # the most accurate match
            embed = self.encode(claim)
            score_batches = [self.similarity_batch(embed, self.attrs_embeddings[i])\
                                                             for i, dataset in enumerate(self.datasets)]
            similarities = [max(batch) for batch in score_batches]
        
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
        else:
            # reset dataset_map[3] to empty list
            for dataset_map in top_k_datasets:
                dataset_map[3] = []

        if verbose: print(f"Most relevant datasets using {method}")
        for dataset_name, _, similarity, relevant_attrs in top_k_datasets:
            if verbose: print(f"Dataset: {dataset_name}, Similarity: {similarity:.2f}, Relevant Attributes: {relevant_attrs}")

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
        

def main():
    matcher = DataMatcher(datasrc="../Datasets")
    # claim = "The energy consumption level of the US was super bad last year."
    # matcher.find_top_k_datasets(claim, k=2)
    phrase1 = "maternal mortality ratio (national estimate, per 100,000 live births)."
    phrase2 = "No country has Child mortality rate higher than 6% in 2011."
    print(matcher.similarity_score("US' economy is larger than China's", "population"))

if __name__ == "__main__":
    main()