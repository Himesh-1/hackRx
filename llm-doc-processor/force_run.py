import sys
import os
import json

# --- Force Path Configuration ---
# This is the most direct way to solve ModuleNotFoundError.
# We are manually adding the necessary paths to the Python interpreter's search path.

print("--- Forcefully modifying Python's search path ---")

# 1. Add the virtual environment's site-packages directory
site_packages_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'venv', 'Lib', 'site-packages'))
if site_packages_path not in sys.path:
    sys.path.insert(0, site_packages_path)
    print(f"Added to path: {site_packages_path}")

# 2. Add the project's root directory to find 'app' and 'models'
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added to path: {project_root}")

print("--- Path modification complete. Attempting to import modules... ---")

try:
    from app import config
    from models.document_loader import DocumentLoader
    from models.chunker import DocumentChunker
    from models.embedder import DocumentEmbedder
    from models.retriever import Retriever
    from models.decision_engine import DecisionEngine
    from models.query_parser import QueryParser
    print("--- All modules imported successfully! ---")
except ImportError as e:
    print(f"\nFATAL ERROR: Failed to import modules even after forcing path: {e}")
    print("This indicates a critical issue with the Python installation or file permissions.")
    sys.exit(1)

def run_pipeline():
    """Executes the end-to-end pipeline test."""
    print("\n--- Starting End-to-End Pipeline Test ---")

    if not config.OPENAI_API_KEY:
        print("\nERROR: OPENAI_API_KEY is not set. Please set it in the .env file.")
        return

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
    run_pipeline()
