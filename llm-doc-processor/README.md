🧠 LLM-Powered Intelligent Query–Retrieval System
================================================

📘 Project Plan & AI Implementation Instructions

This document outlines the complete design, functionality, architecture, and implementation plan for building a high-quality LLM-powered query-response system for large, unstructured documents. This guide will be read by an AI agent to make code changes, so each section includes detailed and clear instructions.

📌 OBJECTIVE
-----------
Design and implement a system that:
- Accepts documents (PDF, DOCX, emails)
- Parses user queries in natural language
- Retrieves relevant clauses based on semantic similarity
- Uses an LLM to generate structured, explainable answers
- Outputs answers in a specified JSON format

Use cases include: insurance policy reading, legal clause extraction, HR and compliance document Q&A.

⚙️ INPUT FORMAT
--------------
Your backend should accept a POST request to `/api/v1/hackrx/run` with the following JSON structure:

{
  "documents": "DOCUMENT_BLOB_URL",
  "questions": [
    "Question 1",
    "Question 2",
    ...
  ]
}

Where:
- documents: A valid blob URL pointing to a PDF, DOCX, or email file (.eml or .msg).
- questions: A list of user queries in natural language.

🎯 OUTPUT FORMAT
---------------
Return a JSON response like this:

{
  "answers": [
    "Answer 1...",
    "Answer 2...",
    ...
  ]
}

Each answer must:
- Be accurate
- Be based strictly on the content of the document
- Include logic from matching clauses (internally used or externally explainable)

🏗️ SYSTEM ARCHITECTURE
-----------------------

1. 📥 Document Loader
   - Download the file from the provided blob URL.
   - Detect file type: .pdf, .docx, .eml, .msg.
   - Use appropriate parser:
     • PDF → PyMuPDF or pdfplumber
     • DOCX → python-docx
     • Email → extract-msg or Python’s email module
   - Extract clean, readable text.
   - Convert raw text into structured, semantically meaningful chunks (e.g., by section, heading, paragraph).

2. 🔍 Chunking & Embedding
   - Split document into intelligent chunks:
     • Prefer splitting at headings, subheadings, or logical paragraph boundaries.
     • Limit chunk size to ~512 tokens (~300–500 words).
   - Generate embeddings for each chunk using:
     • OpenAI's text-embedding-3-small OR
     • Sentence-BERT (e.g., all-MiniLM-L6-v2) via SentenceTransformers
   - Store embeddings in a vector database:
     • Primary: FAISS (for lightweight, local use)
     • Optional: Pinecone (if cloud persistence needed)
   - Assign metadata to each chunk:
     • chunk_id, page_number (if applicable), heading, source_file, etc.

3. ❓ Query Embedding & Retrieval
   - For each question in the input list:
     • Embed the query using the same model as used for document chunks.
     • Perform semantic search in the vector store.
     • Retrieve top-k (e.g., k=3) most similar chunks using cosine similarity.
     • Return matched chunks with:
       - Text content
       - Metadata (e.g., clause ID, page, heading)
       - Similarity score
   - Log retrieval results for traceability.

4. 🧠 Answer Generation (LLM Inference)
   - Use a powerful LLM (e.g., GPT-4) to generate answers.
   - Construct prompt with:
     • Role: "You are a document analysis assistant."
     • Instruction: "Answer based only on the provided clause(s)."
     • Dynamic fields:
       - {user_question}
       - {top_k_clauses} (concatenated relevant clause texts)
   - Prompt Template:

     You are a document analysis assistant. Answer based on the given clause(s) only.

     Question: {user_question}

     Relevant Clause(s):
     {top_k_clauses}

     Answer clearly and concisely.

   - Constraints:
     • Do not hallucinate.
     • If no relevant clause found, respond: "Information not found in the document."
     • Keep answers concise and structured.

5. 📤 JSON Response Generator
   - Map each question to its generated answer.
   - Ensure one answer per question in the output array.
   - Internally log:
     • Input query
     • Retrieved clauses
     • LLM input/output
     • Final answer
   - Return final response in strict JSON format.

🧪 TESTING & SAMPLE DATA
------------------------
Sample input:

{
  "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...",
  "questions": [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "What is the No Claim Discount (NCD) offered in this policy?",
    "Is there a benefit for preventive health check-ups?",
    "How does the policy define a 'Hospital'?",
    "What is the extent of coverage for AYUSH treatments?",
    "Are there any sub-limits on room rent and ICU charges for Plan A?"
  ]
}

sample output:

{
"answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
        "The policy has a specific waiting period of two (2) years for cataract surgery.",
        "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
        "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
        "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
        "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
        "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
        "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
    ]
}


Verify responses against expected answers in the HackRx problem statement.

✅ FUNCTIONAL REQUIREMENTS
--------------------------
| Feature                   | Required? | Notes                                      |
|---------------------------|-----------|--------------------------------------------|
| PDF parsing               | ✅        | Use PyMuPDF or pdfplumber                  |
| DOCX parsing              | ✅        | Use python-docx                            |
| Email parsing             | ✅        | Use extract-msg or email module            |
| Semantic chunking         | ✅        | Heading-wise or paragraph-wise             |
| Embedding-based retrieval | ✅        | Use FAISS or Pinecone                      |
| Natural language understanding | ✅  | Via GPT-4 or equivalent LLM                |
| Explainable output        | ✅        | Track clause sources and scores            |
| JSON formatted answers    | ✅        | One answer per query                       |

⚡ NON-FUNCTIONAL REQUIREMENTS
------------------------------
- Token Efficiency: Use only top-k retrieved chunks (e.g., k=3) in LLM context.
- Low Latency: Aim for <5 seconds per query. Use async processing where possible.
- Reusability: Each module in a separate Python file/class.
- Explainability: Log clause matches, similarity scores, and full LLM prompts/responses.
- Security: Validate and sanitize URLs; avoid arbitrary code execution.
- Modularity: Components must be swappable (e.g., change LLM or vector DB easily).

🧠 RECOMMENDED STACK
--------------------
| Component             | Technology                              |
|-----------------------|------------------------------------------|
| Backend API           | FastAPI                                  |
| LLM                   | OpenAI GPT-4                             |
| Embedding Model       | text-embedding-3-small or all-MiniLM-L6-v2 |
| Vector Store          | FAISS (default), Pinecone (optional)     |
| PDF Parsing           | PyMuPDF or pdfplumber                    |
| DOCX Parsing          | python-docx                              |
| Email Parsing         | extract-msg or email module              |
| Data Format           | JSON                                     |
| Optional DB           | PostgreSQL / SQLite (for logs/scoring)   |

🧠 BONUS: ADVANCED FEATURES (Optional)
-------------------------------------
- Support multilingual documents via translation API (e.g., Google Translate)
- Add simple UI to input query and highlight source clause
- Return confidence score with each answer (based on similarity + LLM certainty)
- Persistent document indexing using Pinecone/Weaviate
- Redis caching for repeated queries
- Rate limiting and request validation

🚀 HOW AI SHOULD INTERPRET THIS PLAN
-----------------------------------
- Create one file/module per component:
  • loader.py       → Handles downloading and file detection
  • parser.py       → Extracts and cleans text per file type
  • chunker.py      → Splits text into semantic chunks
  • embedder.py     → Generates embeddings and stores in FAISS
  • retriever.py    → Handles query embedding and similarity search
  • llm_answer.py   → Calls LLM and formats response
  • api.py          → FastAPI endpoint handler
  • utils.py        → Shared helpers (logging, sanitization, etc.)

- Comment every function:
  • Input type
  • Output type
  • Purpose

- Ensure the system runs as a single FastAPI app:
  uvicorn api:app --reload

- Maintain separation between business logic and I/O operations.

- Avoid hardcoding:
  • Use environment variables for API keys, URLs, chunk size, k-value, etc.

📊 SCORING SYSTEM NOTES
----------------------
Follow HackRx scoring rules:
- Unknown documents → 2.0x weight
- Known documents → 0.5x weight
- Each question has individual weight (e.g., 1.0 or 2.0)
- Accuracy + traceability = higher score

Ensure every answer is:
- Traceable (linked to clause)
- Correct
- Efficient (low latency, minimal tokens)

📦 DEPLOYMENT INSTRUCTIONS
-------------------------
1. Create `.env` file:
   OPENAI_API_KEY=your_key_here
   CHUNK_SIZE=512
   TOP_K=3
   VECTOR_STORE=faiss  # or pinecone

2. Install dependencies:
   pip install -r requirements.txt

3. Run server:
   uvicorn api:app --reload

4. Test with curl:
   curl -X POST http://localhost:8000/api/v1/hackrx/run \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...",
       "questions": [
         "What is the grace period for premium payment?"
       ]
     }'

5. Validate output format and correctness.

🏁 FINAL CHECKLIST BEFORE SUBMISSION
------------------------------------
✅ All 3 file types supported: PDF, DOCX, email
✅ Logs contain clause match info (text, score, metadata)
✅ Output is explainable, correct, and JSON-formatted
✅ Latency acceptable (< 5 seconds per query)
✅ API tested with Postman/curl
✅ Code is modular, commented, and clean
✅ No hardcoded secrets or paths
✅ Requirements.txt includes all dependencies
✅ .env used for configuration
✅ Error handling for invalid URLs, unsupported formats, empty results