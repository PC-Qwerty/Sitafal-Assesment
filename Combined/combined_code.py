import os
from typing import List, Dict, Tuple
import requests
from bs4 import BeautifulSoup
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Step 1: Load PDF
def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Step 2: Scrape Website Content
def scrape_website(urls: List[str]):
    loader = WebBaseLoader(urls)
    documents = loader.load()
    return documents

# Step 3: Split Text into Chunks
def split_text(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks

# Step 4: Create FAISS Index
def create_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Step 5: Set up Retrieval-based Question Answering
def setup_retrieval_qa(vectorstore):
    llm = OllamaLLM(model="llama2:7b")  # Replace with the desired Ollama model
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain

# Step 6: Query Handling
def query_documents(qa_chain, query: str):
    response = qa_chain.run(query)
    return response

# Step 7: Save Results to a File
def save_results_to_file(file_path: str, results: Dict, metadata: Dict = None):
    """
    Save query results to a file for review.

    Args:
        file_path (str): Path to the results file.
        results (Dict): Dictionary of query results.
        metadata (Dict): Optional metadata for additional context (e.g., URLs or source info).
    """
    with open(file_path, "a", encoding="utf-8") as file:
        for query, response in results.items():
            file.write(f"Query: {query}\n")
            if metadata and query in metadata:
                file.write(f"Source: {metadata[query]}\n")
            file.write(f"Response: {response}\n\n")

# Example Usage for PDF and Website Integration
if __name__ == "__main__":
    # Example for PDFs
    pdf_path = "C:/Users/Pavan/Desktop/Sitafal/Task1/pdf.pdf"
    pdf_results_file = "C:/Users/Pavan/Desktop/Sitafal/Combined/pdf_results.txt"

    # Load and process PDF
    pdf_documents = load_pdf(pdf_path)
    pdf_chunks = split_text(pdf_documents)
    pdf_vectorstore = create_faiss_index(pdf_chunks)
    pdf_qa_chain = setup_retrieval_qa(pdf_vectorstore)

    pdf_queries = [
        "What types of visual data representations are discussed in the PDF?",
        "What is the GDP of the manufacturing sector in 2015?"
    ]

    pdf_results = {}
    for query in pdf_queries:
        response = query_documents(pdf_qa_chain, query)
        pdf_results[query] = response

    save_results_to_file(pdf_results_file, pdf_results, metadata={"Source": pdf_path})
    print(f"PDF results saved to {pdf_results_file}")

    # Example for Websites
    urls = [
        "https://www.uchicago.edu/",
        "https://www.washington.edu/",
        "https://www.stanford.edu/",
        "https://und.edu/"
    ]
    website_results_file = "C:/Users/Pavan/Desktop/Sitafal/Combined/website_results.txt"

    # Load and process websites
    website_documents = scrape_website(urls)
    website_chunks = split_text(website_documents)
    website_vectorstore = create_faiss_index(website_chunks)
    website_qa_chain = setup_retrieval_qa(website_vectorstore)

    website_queries = [
        "What is the history of the University of Chicago?",
        "What programs does Stanford offer?"
    ]

    website_results = {}
    for query in website_queries:
        response = query_documents(website_qa_chain, query)
        website_results[query] = response

    save_results_to_file(website_results_file, website_results, metadata={url: f"Content from {url}" for url in urls})
    print(f"Website results saved to {website_results_file}")
