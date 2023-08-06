from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json

class DataMatcher:
    def __init__(self, datasrc: str = None):
        self.datasrc = datasrc

    def find_top_k_datasets(self, claim: str, k: int = 2):
        # Load a pre-trained SentenceTransformer model
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # load datasets description
        with open(f'{self.datasrc}/description/desc.json', 'r') as openfile:
            datasets = json.load(openfile)

        # Compute embeddings for the text and dataset descriptions
        text_embedding = model.encode([claim])[0]
        dataset_embeddings = model.encode([dataset['description'] for dataset in datasets])

        # Compute cosine similarity between the text and dataset descriptions
        similarities = cosine_similarity([text_embedding], dataset_embeddings)[0]
        # Combine the dataset names, descriptions, and their corresponding similarities
        result = [(dataset['name'], dataset['description'], similarity) for dataset, similarity in zip(datasets, similarities)]
        # Sort the result based on similarity in descending order
        result.sort(key=lambda x: x[2], reverse=True)

        top_k_datasets = result[:k]

        print("Most relevant datasets:")
        for dataset_name, _, similarity in top_k_datasets:
            print(f"Dataset: {dataset_name}, Similarity: {similarity:.2f}")

        return top_k_datasets

if __name__ == "__main__":
    matcher = DataMatcher(datasrc="../Datasets")
    claim = "The energy consumption level of the US was super bad last year."
    print(matcher.find_top_k_datasets(claim, k=2))