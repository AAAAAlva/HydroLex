import os
import re
import json
import time
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
from smolagents import OpenAIServerModel

# --- 1. GLOBAL CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# System Paths
INPUT_DOCS_DIR = Path("./data/documents")          # Target folder with source files
BASE_TARGET_DIR = Path("./output")                 # Root folder for database & outputs
DB_SAVE_PATH = BASE_TARGET_DIR / "vector_db"       # FAISS vector database directory

EXCEL_INPUT_PATH = "./data/questions.xlsx"         # Input Excel file with evaluation questions
EXCEL_OUTPUT_PATH = BASE_TARGET_DIR / "rag_results.xlsx"  # Output Excel file for RAG results

# Ollama Endpoint
OLLAMA_API_BASE = "http://localhost:11434/v1"      # URL where Ollama is hosted


# Hyperparameters
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
MODEL_NAME = "llama3.1"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".txt"}
MAX_RETRIEVAL_LOOPS = 10
TOP_K_RETRIEVAL = 5


# --- 2. VECTOR DATABASE INGESTION & LOADING ---
def build_vector_db():
    """Scans local documents, chunks content, and builds a FAISS vector index."""
    logger.info(f"Scanning target directory recursively: {INPUT_DOCS_DIR}")
    
    all_file_paths = [
        f for f in INPUT_DOCS_DIR.rglob("*")
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    logger.info(f"Discovered {len(all_file_paths)} supported document files.")

    source_docs = []
    for file_path in tqdm(all_file_paths, desc="Converting documents via Docling"):
        try:
            loader = DoclingLoader(file_path=str(file_path), export_type=ExportType.MARKDOWN)
            loaded_pages = loader.load()
            clean_title = file_path.stem.replace("_", " ").title()
            
            for doc in loaded_pages:
                doc.metadata["title"] = doc.metadata.get("title", clean_title)
                doc.metadata["source"] = str(file_path)
                doc.metadata.setdefault("page", 1)
                
            source_docs.extend(loaded_pages)
        except Exception as e:
            logger.error(f"Failed to convert {file_path.name}: {e}")

    # Chunking & Deduplication
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    logger.info("Splitting layouts into chunks...")
    docs_processed = []
    seen_texts = set()

    for doc in tqdm(source_docs, desc="Splitting documents"):
        for chunk in text_splitter.split_documents([doc]):
            if chunk.page_content not in seen_texts:
                seen_texts.add(chunk.page_content)
                docs_processed.append(chunk)

    logger.info(f"Created {len(docs_processed)} unique chunks.")

    # Vector Index Creation
    logger.info("Generating embeddings and building FAISS index...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = FAISS.from_documents(documents=docs_processed, embedding=embedding_model)
    
    BASE_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(DB_SAVE_PATH))
    logger.info(f"FAISS database successfully saved to: {DB_SAVE_PATH}")
    return vectordb


def load_vector_db():
    """Loads existing FAISS vector database or builds it if missing."""
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if DB_SAVE_PATH.exists():
        logger.info(f"Loading vector database from: {DB_SAVE_PATH}")
        return FAISS.load_local(
            str(DB_SAVE_PATH),
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        logger.info("Vector database not found. Starting ingestion build...")
        return build_vector_db()


# --- 3. HELPER FUNCTIONS FOR AGENT LOOP ---
def run_retriever_step(vectordb, query: str, all_retrieved_chunks: list) -> str:
    """Performs vector search and appends unique chunks to the master tracking list."""
    results = vectordb.similarity_search(query, k=TOP_K_RETRIEVAL)
    step_chunks = []
    
    for doc in results:
        step_chunks.append(doc.page_content)
        if doc.page_content not in all_retrieved_chunks:
            all_retrieved_chunks.append(doc.page_content)
            
    return "\n\n".join([f"Snippet {i+1}: {c}" for i, c in enumerate(step_chunks)])


def call_llm(model, prompt: str) -> str:
    """Helper wrapper for clean LLM completions."""
    response = model([{"role": "user", "content": prompt}])
    return response.content if hasattr(response, "content") else str(response)


# --- 4. MAIN AGENTIC RAG EXECUTION ---
def main():
    vectordb = load_vector_db()
    
    model = OpenAIServerModel(
        model_id=MODEL_NAME,
        api_base=OLLAMA_API_BASE,
        api_key="ollama",
        temperature=0.0
    )

    df_input = pd.read_excel(EXCEL_INPUT_PATH, sheet_name="Sheet1")
    questions = df_input["Original Question"].dropna().tolist()
    logger.info(f"Loaded {len(questions)} target questions.")

    # Checkpoint setup
    results_records = []
    processed_questions = set()
    
    if os.path.exists(EXCEL_OUTPUT_PATH):
        try:
            df_existing = pd.read_excel(EXCEL_OUTPUT_PATH)
            results_records = df_existing.to_dict(orient="records")
            processed_questions = set(df_existing["Question"].tolist())
            logger.info(f"Checkpoint loaded. Resuming from row {len(results_records)}.")
        except Exception as e:
            logger.warning(f"Could not read existing output file. Starting fresh. Error: {e}")

    # Iterative Evaluation Execution Loop
    for idx, question in enumerate(questions):
        if question in processed_questions:
            continue

        logger.info(f"[{idx+1}/{len(questions)}] Processing Question: {question}")
        
        all_retrieved_chunks = []
        last_step_chunks = "None"
        generated_queries = []
        agent_thoughts = []
        
        keep_searching = True
        current_loop = 1

        while keep_searching and current_loop <= MAX_RETRIEVAL_LOOPS:
            agent_thoughts.append(f"[Step {current_loop}] Starting retrieval attempt.")
            
            # Step 1: Formulate Search Query
            context_preview = " None" if not all_retrieved_chunks else " ".join(all_retrieved_chunks[:2])
            query_prompt = f"""Based on the user question, generate a concise search query string to search the database. 
Output ONLY the query string without quotes or punctuation.

User Question: {question}
Context Collected So Far: {context_preview}"""
            
            search_query = call_llm(model, query_prompt).strip().strip('"').strip("'")
            generated_queries.append(f"Query {current_loop}: {search_query}")
            logger.info(f"Loop {current_loop}: Searching DB for -> '{search_query}'")

            # Step 2: Retrieve Relevant Passages
            try:
                last_step_chunks = run_retriever_step(vectordb, search_query, all_retrieved_chunks)
            except Exception as e:
                logger.error(f"Retrieval error: {e}")
                last_step_chunks = f"Error during retrieval: {e}"

            # Step 3: Judge Sufficiency Over ALL Accumulated Chunks
            accumulated_context = "\n\n".join([f"Snippet {i+1}: {c}" for i, c in enumerate(all_retrieved_chunks)])
            
            judgment_prompt = f"""Analyze ALL retrieved context snippets relative to the target question. 
Determine if this context contains sufficient details to fully and accurately answer the question.
Reply strictly in JSON with keys: "sufficient" (boolean) and "reasoning" (string).

Target Question: {question}
Accumulated Context:
{accumulated_context}

Output JSON:"""
            
            judge_raw = call_llm(model, judgment_prompt)
            agent_thoughts.append(f"[Step {current_loop} Judge Monologue]\n{judge_raw.strip()}")

            # Step 4: Parse Judgment
            try:
                clean_json_str = re.sub(r"```json|```", "", judge_raw).strip()
                judge_data = json.loads(clean_json_str)
                if judge_data.get("sufficient") is True:
                    logger.info(f"Loop {current_loop}: Context determined sufficient. Stopping search.")
                    keep_searching = False
            except Exception:
                if "true" in judge_raw.lower():
                    logger.info(f"Loop {current_loop}: Context determined sufficient via fallback parser.")
                    keep_searching = False

            current_loop += 1

        # Step 5: Final Answer Generation
        logger.info("Generating Final Grounded Answer...")
        final_references = "\n\n".join(all_retrieved_chunks) if all_retrieved_chunks else "No relevant context found."
        
        final_prompt = f"""Answer the following question accurately using ONLY the verified regulatory references below.
If the references do not contain enough information, state what is missing clearly.

Question: {question}
Verified Regulatory References:
{final_references}"""

        final_answer = call_llm(model, final_prompt)

        # Log & Cache Progress
        results_records.append({
            "Question": question,
            "Final Answer": final_answer,
            "All Context Chunks": "\n---\n".join(all_retrieved_chunks),
            "Last Step Context": last_step_chunks,
            "Steps Taken": len(generated_queries),
            "Queries Tried": " | ".join(generated_queries),
            "Agent Monologue": "\n\n".join(agent_thoughts)
        })

        try:
            pd.DataFrame(results_records).to_excel(EXCEL_OUTPUT_PATH, index=False)
            logger.info(f"Saved progress for question {idx+1} to disk.")
        except Exception as e:
            logger.error(f"Failed to auto-save progress: {e}")

        time.sleep(1)

    logger.info(f"Process finished! Results saved to: {EXCEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()