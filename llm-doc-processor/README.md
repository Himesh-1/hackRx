# llm-doc-processor

Use LLM + semantic search to process real-world insurance/legal queries against large documents.

## Goal
This project enables natural language queries (e.g., "46-year-old male, knee surgery in Pune, 3-month-old insurance policy") to be parsed, relevant clauses retrieved from unstructured documents (PDFs, DOCXs), and a decision (approve/reject, payout) made using LLMs and vector search. The system returns a structured JSON response with decision, amount, and justification referencing document clauses.
