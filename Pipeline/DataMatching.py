from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Summarizer import Summarizer
import pandas as pd
import json
import os

class DataMatcher:
    def __init__(self, datasrc: str=None):
        self.summarizer = Summarizer(datasrc=datasrc)
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # prepare sentence embedder
        
        if datasrc:
            self.datasrc = datasrc
            self.datasets = [filename for filename in os.listdir(self.datasrc) if filename.endswith('.csv')]

            with open(f'{self.datasrc}/description/desc.json', 'r+') as openfile:
                self.description = json.load(openfile)

                # prepare dataset embeddings
                writeflag = False
                for dataset in self.datasets:
                    if dataset not in self.description:
                        self.description[dataset] = self.summarizer.summarize(dataset)
                        writeflag = True
                self.dataset_embeddings = self.embedder.encode([self.description[dataset] for dataset in self.datasets])

                # write summaries back to file
                if writeflag:
                    openfile.seek(0)
                    json.dump(self.description, openfile)
                openfile.close()

    def find_top_k_datasets(self, claim: str, k: int = 2, method: str = "attr"):
        assert self.datasrc, "Datasrc not specified."

        # Compute embeddings for the text and dataset descriptions
        if method == "cosine":
            similarities = self.similarity_batch(claim, self.dataset_embeddings)
        elif method == "idf":
            similarities = [self.idf_score(claim, self.description[dataset]['description']) for dataset in self.datasets]
        elif method == "attr":
            similarities = []
            for dataset in self.datasets:
                df = pd.read_csv(f"{self.datasrc}/{dataset}")
                attributes = list(df.columns)
                similarities.append(self.attr_score(claim, attributes))
        
        # Combine the dataset names, descriptions, and their corresponding similarities
        result = [(self.description[name]['name'], self.description[name]['description'], similarity) \
                            for name, similarity in zip(self.datasets, similarities)]
        # Sort the result based on similarity in descending order
        result.sort(key=lambda x: x[2], reverse=True)

        top_k_datasets = result[:k]

        print(f"Most relevant datasets using {method}")
        for dataset_name, _, similarity in top_k_datasets:
            print(f"Dataset: {dataset_name}, Similarity: {similarity:.2f}")

        return top_k_datasets

    def similarity_score(self, phrase1: str, phrase2: str):
        phrase1_embedding = self.embedder.encode([phrase1])[0]
        phrase2_embedding = self.embedder.encode([phrase2])[0]
        similarity = cosine_similarity([phrase1_embedding], [phrase2_embedding])[0][0]
        return similarity
    
    def similarity_batch(self, phrase: str, batch_of_phrases):
        phrase_embedding = self.embedder.encode([phrase])[0]
        if isinstance(batch_of_phrases[0], str):
            batch_of_embeddings = self.embedder.encode(batch_of_phrases)
        else: # embeddings are already computed
            batch_of_embeddings = batch_of_phrases
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
    
    def attr_score(self, phrase: str, attributes: list):
        # Open the datasrc
        return max(self.similarity_batch(phrase, attributes))
        

if __name__ == "__main__":
    matcher = DataMatcher(datasrc="../Datasets")
    # claim = "The energy consumption level of the US was super bad last year."
    # matcher.find_top_k_datasets(claim, k=2)
    phrase1 = "Population age has been decreasing."
    phrase2 = "Educational attainment, at least completed post-secondary, population 25+, female (%) (cumulative)"
    matcher.find_top_k_datasets(phrase1, k=5, method="attr")