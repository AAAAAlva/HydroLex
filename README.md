# HydroLex
HydroLex is an LLM-guided Agentic Retrieval-Augmented Generation (RAG) system for environmental regulatory question answering, with an initial focus on U.S. water quality regulations and nutrient management policies.
The system integrates document processing, semantic retrieval, and iterative LLM reasoning to identify relevant regulatory information and generate evidence-grounded answers.

# What it does
- **Regulatory Q&A** — Answers environmental policy questions using retrieved evidence from regulatory documents and supporting references.
- **Automated knowledge base construction** — Processes regulatory documents, generates text embeddings, and builds a searchable FAISS vector database.
- **LLM-guided query generation** — Uses an LLM to formulate search queries based on the user's question and previously retrieved context.
- **Iterative retrieval** — Repeatedly searches the knowledge base to collect additional evidence, with a maximum of 10 retrieval iterations.
- **Evidence sufficiency assessment — Uses an LLM to evaluate whether accumulated evidence is sufficient to answer the question, allowing retrieval to stop early.
- **Traceable evaluation outputs** — Records generated queries, retrieved passages, retrieval iterations, LLM judgments, and final answers for subsequent analysis.

# Installation
- Install the required Python dependencies:
```bash
pip install \
    pandas \
    openpyxl \
    tqdm \
    docling \
    langchain-docling \
    langchain-text-splitters \
    langchain-community \
    langchain-huggingface \
    faiss-cpu \
    sentence-transformers \
    transformers \
    smolagents
```
The current implementation uses Ollama to serve Llama 3.1 locally.

- Install Ollama and download the model and ensure that the Ollama server is running:
- ```bash
  ollama pull llama3.1
  ollama serve
  ```


# Configuration reference
The default configuration in [Ag_llama.py](https://github.com/AAAAAlva/HydroLex/blob/main/Ag_llama.py) is:
```bash
OLLAMA_API_BASE = "your_local_host"
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
MODEL_NAME = "llama3.1"
MAX_RETRIEVAL_LOOPS = 10
TOP_K_RETRIEVAL = 5
```
  
