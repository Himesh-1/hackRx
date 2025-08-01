
"""
End-to-End Test Script
This script runs the entire document processing pipeline without starting the web server.

Prerequisite: Run `setup.py` first to ensure all dependencies are installed.
"""

import os
import sys
import json

# --- Setup Python Path ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# --- Import Core Components ---
from app import config
from models.document_loader import DocumentLoader
from models.chunker import DocumentChunker
from models.embedder import DocumentEmbedder
from models.retriever import Retriever
from models.decision_engine import DecisionEngine
from models.query_parser import QueryParser

def run_test():
    """Executes the end-to-end pipeline test."""
    print("--- Starting End-to-End Pipeline Test ---")

    if not config.OPENAI_API_KEY:
        print("\nERROR: OPENAI_API_KEY is not set. Please set it in the .env file.")
        return

    # ... (The rest of the script remains the same) ...
    print("\nStep 1: Loading documents...")
    loader = DocumentLoader(docs_path=str(config.DOCS_DIR))
    documents = loader.load_all_documents()
    print(f"Loaded {len(documents)} document(s).")

    print("\nStep 2: Chunking documents...")
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("\nStep 3: Embedding chunks...")
    embedder = DocumentEmbedder(
        model_name=config.EMBEDDING_MODEL,
        embedding_provider=config.EMBEDDING_PROVIDER,
        cache_dir=str(config.EMBEDDINGS_DIR)
    )
    embedded_chunks = embedder.embed_chunks(chunks)
    print(f"Created {len(embedded_chunks)} embeddings.")

    print("\nStep 4: Initializing retriever...")
    retriever = Retriever(embedder, embedded_chunks)
    print("Retriever initialized.")

    print("\nStep 5: Initializing decision engine...")
    decision_engine = DecisionEngine(model_name=config.DECISION_ENGINE_MODEL)
    print("Decision engine initialized.")

    test_query = "I have a 3-month old policy. Is my knee surgery covered?"
    print(f"\nStep 6: Processing test query: '{test_query}'")
    query_parser = QueryParser()
    parsed_query = query_parser.parse_query(test_query)
    print(f"Parsed entities: {parsed_query.entities}")

    print("\nStep 7: Retrieving relevant document chunks...")
    retrieved_chunks = retriever.retrieve(parsed_query, top_k=config.TOP_K_RETRIEVAL)
    print(f"Retrieved {len(retrieved_chunks)} chunks.")

    print("\nStep 8: Making final decision...")
    final_decision = decision_engine.make_decision(parsed_query, retrieved_chunks)

    print("\n--- FINAL DECISION ---")
    print(json.dumps(final_decision, indent=2))
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    run_test()
