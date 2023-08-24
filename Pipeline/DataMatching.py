from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Summarizer import Summarizer
import json
import os

class DataMatcher:
    def __init__(self, datasrc: str=None):
        self.summarizer = Summarizer(datasrc=datasrc)
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # prepare sentence embedder
        
        if datasrc: # only if datasrc is provided
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

    def find_top_k_datasets(self, claim: str, k: int = 2):
        # Compute embeddings for the text and dataset descriptions
        text_embedding = self.embedder.encode([claim])[0]

        # Compute cosine similarity between the text and dataset descriptions
        similarities = cosine_similarity([text_embedding], self.dataset_embeddings)[0]
        # Combine the dataset names, descriptions, and their corresponding similarities
        result = [(self.description[name]['name'], self.description[name]['description'], similarity) \
                            for name, similarity in zip(self.datasets, similarities)]
        # Sort the result based on similarity in descending order
        result.sort(key=lambda x: x[2], reverse=True)

        top_k_datasets = result[:k]

        print("Most relevant datasets:")
        for dataset_name, _, similarity in top_k_datasets:
            print(f"Dataset: {dataset_name}, Similarity: {similarity:.2f}")

        return top_k_datasets

    def similarity_score(self, phrase1: str, phrase2: str):
        phrase1_embedding = self.embedder.encode([phrase1])[0]
        phrase2_embedding = self.embedder.encode([phrase2])[0]
        similarity = cosine_similarity([phrase1_embedding], [phrase2_embedding])[0][0]
        return similarity

if __name__ == "__main__":
    matcher = DataMatcher(datasrc="../Datasets")
    # claim = "The energy consumption level of the US was super bad last year."
    # matcher.find_top_k_datasets(claim, k=2)
    phrase1 = "2011"
    phrase2 = "Educational attainment, at least completed post-secondary, population 25+, female (%) (cumulative)"
    print(matcher.similarity_score(phrase1, phrase2))