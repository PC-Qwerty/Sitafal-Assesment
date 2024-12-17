import os
from typing import List, Dict, Tuple
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Extract Text from PDFs
def extract_text_from_pdf(file_path: str) -> Dict[int, str]:
    text_by_page = {}
    
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_by_page[i + 1] = text

    return text_by_page

# Step 2: Extract Tabular Data
def extract_tables_from_pdf(file_path: str) -> Dict[int, List]:
    tables_by_page = {}

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if tables:
                tables_by_page[i + 1] = tables

    return tables_by_page

# Step 3: Chunking the Text
def chunk_text(text_by_page: Dict[int, str], chunk_size: int = 100) -> List[str]:
    chunks = []
    for page, text in text_by_page.items():
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# Step 4: Generate Embeddings
def generate_embeddings(chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> Tuple[List[str], np.ndarray]:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return chunks, np.array(embeddings)

# Step 5: Create and Search FAISS Index
def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Dimension of embeddings
    index.add(embeddings)
    return index

def search_faiss_index(index: faiss.IndexFlatL2, query_embedding: np.ndarray, top_k: int = 5) -> List[int]:
    distances, indices = index.search(query_embedding, top_k)
    return indices[0].tolist()

# Step 6: Query Handling
def handle_query(query: str, chunks: List[str], index: faiss.IndexFlatL2, model_name: str = "all-MiniLM-L6-v2") -> List[str]:
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], convert_to_tensor=False)
    top_indices = search_faiss_index(index, query_embedding)
    return [chunks[i] for i in top_indices]

# Step 7: Save Results to a File
def save_results_to_file(file_path: str, results: Dict, metadata: Dict = None, tables: Dict[int, List] = None):
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

        if tables:
            file.write("Extracted Tables:\n")
            for page, table in tables.items():
                file.write(f"Page {page}:\n")
                for row in table:
                    file.write(f"{row}\n")
                file.write("\n")

# Example Usage
if __name__ == "__main__":
    pdf_path = "C:/Users/Pavan/Desktop/Sitafal/Task1/pdf.pdf"
    results_file = "C:/Users/Pavan/Desktop/Sitafal/Task1/results.txt"

    # Step 1: Extract text
    text_data = extract_text_from_pdf(pdf_path)
    print("Extracted Text:", text_data)

    # Step 2: Extract tables
    table_data = extract_tables_from_pdf(pdf_path)
    print("Extracted Tables:", table_data)

    # Step 3: Chunk text
    chunks = chunk_text(text_data)
    print("Text Chunks:", chunks)

    # Step 4: Generate embeddings
    chunks, embeddings = generate_embeddings(chunks)
    print("Generated Embeddings:", embeddings.shape)

    # Step 5: Create FAISS index
    faiss_index = create_faiss_index(embeddings)

    # Step 6: Query Handling
    user_queries = {
        "What types of visual data representations are discussed in the PDF?": "Pie charts, bar graphs, and line graphs.",
        "What is the GDP of the manufacturing sector in 2015?": "5,829,554 million dollars.",
    }

    query_results = {}
    for query in user_queries.keys():
        relevant_chunks = handle_query(query, chunks, faiss_index)
        query_results[query] = relevant_chunks

    # Step 7: Save results to file
    save_results_to_file(results_file, query_results, metadata={"PDF Source": pdf_path}, tables=table_data)

    print(f"Results saved to {results_file}")
