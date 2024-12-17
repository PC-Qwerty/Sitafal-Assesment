import os
from typing import List, Dict, Tuple
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Crawl and Scrape Website Content
def scrape_website(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve content from {url}")

    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract textual content
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text

# Step 2: Chunking Website Content
def chunk_text(text: str, chunk_size: int = 100) -> List[str]:
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Step 3: Generate Embeddings
def generate_embeddings(chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> Tuple[List[str], np.ndarray]:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return chunks, np.array(embeddings)

# Step 4: Create and Search FAISS Index
def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Dimension of embeddings
    index.add(embeddings)
    return index

def search_faiss_index(index: faiss.IndexFlatL2, query_embedding: np.ndarray, top_k: int = 5) -> List[int]:
    distances, indices = index.search(query_embedding, top_k)
    return indices[0].tolist()

# Step 5: Query Handling
def handle_query(query: str, chunks: List[str], index: faiss.IndexFlatL2, model_name: str = "all-MiniLM-L6-v2") -> List[str]:
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], convert_to_tensor=False)
    top_indices = search_faiss_index(index, query_embedding)
    return [chunks[i] for i in top_indices]

# Step 6: Save Results to a File
def save_results_to_file(file_path: str, results: Dict, metadata: Dict = None):
    with open(file_path, "a", encoding="utf-8") as file:
        for query, response in results.items():
            file.write(f"Query: {query}\n")
            if metadata and query in metadata:
                file.write(f"Source: {metadata[query]}\n")
            if isinstance(response, list):
                for line in response:
                    file.write(f"- {line}\n")
            else:
                file.write(f"- {response}\n")
            file.write("\n")

# Example Usage
if __name__ == "__main__":
    urls = [
        "https://www.uchicago.edu/",
        "https://www.washington.edu/",
        "https://www.stanford.edu/",
        "https://und.edu/"
    ]

    results_file = "C:/Users/Pavan/Desktop/Sitafal/Task2/website_results.txt"

    # Step 1: Scrape websites
    all_text = ""
    metadata = {}
    for url in urls:
        try:
            text = scrape_website(url)
            all_text += text
            metadata[url] = f"Content from {url}"
        except ValueError as e:
            print(f"Error scraping {url}: {e}")

    # Step 2: Chunk text
    chunks = chunk_text(all_text)
    print("Text Chunks:", chunks)

    # Step 3: Generate embeddings
    chunks, embeddings = generate_embeddings(chunks)
    print("Generated Embeddings:", embeddings.shape)

    # Step 4: Create FAISS index
    faiss_index = create_faiss_index(embeddings)

    # Step 5: Query Handling
    user_queries = {
        "What is the history of the University of Chicago?": "Expected output or description.",
        "What programs does Stanford offer?": "Expected output or description."
    }

    query_results = {}
    for query in user_queries.keys():
        relevant_chunks = handle_query(query, chunks, faiss_index)
        query_results[query] = relevant_chunks

    # Step 6: Save results to file
    save_results_to_file(results_file, query_results, metadata)

    print(f"Results saved to {results_file}")
