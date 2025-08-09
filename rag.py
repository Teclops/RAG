# app.py - Complete HackRX RAG System

from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import requests
import tempfile
from pathlib import Path
from typing import List, Dict
import json
import time
import uvicorn

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_nomic import NomicEmbeddings

# Set API key (from https://atlas.nomic.ai)
os.environ["NOMIC_API_KEY"] = "nk-5Kl9IcLShbLzOWhZFZ7Lhe39FEpYTwJnF12OVceGC04"

# ================= FastAPI App Setup =================
app = FastAPI(
    title="HackRX RAG System",
    description="LLM-Powered Document Query System for Insurance Policies",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
EXPECTED_API_KEY = "88d4175f44cb52246172bfffb2c66b1659c3c8aca12d6a16a5ba15a9cd679e3c"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token authentication"""
    if credentials.credentials != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

# ================= Request/Response Models =================
class HackRXRequest(BaseModel):
    """Request model for /hackrx/run endpoint"""
    documents: str = Field(..., description="URL to the PDF document")
    questions: List[str] = Field(..., description="List of questions to answer")

class HackRXResponse(BaseModel):
    """Response model for /hackrx/run endpoint"""
    answers: List[str] = Field(..., description="List of answers corresponding to questions")

# ================= Document Processing Functions =================

async def download_pdf_from_url(url: str) -> str:
    """Download PDF from URL and return temporary file path"""
    try:
        print(f"[INFO] Downloading PDF from: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=60, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
            tmp_file_path = tmp_file.name
        
        print(f"[INFO] PDF downloaded successfully to: {tmp_file_path}")
        return tmp_file_path
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF download: {str(e)}")

def load_and_process_pdf(pdf_path: str) -> List:
    """Load PDF and extract documents"""
    try:
        print(f"[INFO] Loading PDF from: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"[INFO] Successfully loaded {len(documents)} pages")
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load PDF: {str(e)}")

def create_vectorstore(documents: List) -> FAISS:
    """Create FAISS vectorstore from documents"""
    try:
        print(f"[INFO] Creating vectorstore from {len(documents)} documents")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )
        
        texts = text_splitter.split_documents(documents)
        print(f"[INFO] Created {len(texts)} text chunks")
        
        # Create embeddings instance
        embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
        
        vectorstore = FAISS.from_documents(texts, embeddings)
        print(f"[INFO] Vectorstore created successfully")
        return vectorstore
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create vectorstore: {str(e)}")

def create_rag_chain(vectorstore: FAISS):
    """Create RAG chain for precise, clause-backed insurance QA"""
    try:
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for better coverage
            search_kwargs={
                "k": 8,          # final chunks passed to LLM (lower for precision)
                "fetch_k": 30,   # candidates considered
                "lambda_mult": 0.75  # balance relevance vs diversity
            }
        )

        llm = ChatGoogleGenerativeAI(
            api_key="AIzaSyArmeI1xitvuicUDw9vAvQj467yFeBTMWE",
            model="gemini-2.0-flash",
            temperature=0.0,    # absolutely factual
            max_tokens=1500,    # avoid long rambles
            top_p=0.8
        )

        prompt_template = """You are an expert insurance policy analyst.
Answer ONLY from the provided policy context — never use outside knowledge.

Context from Policy Document:
{context}

Question:
{question}

Output Rules:
1. Format as a **concise 'Policy Breakdown'** with clear bold section headers.
2. Each main rule: top-level bullet (`-`).
3. Sub-points: 2-space indent before the dash.
4. ALWAYS include clause numbers exactly as shown (e.g., "(Clause 13.i)").
5. Preserve exact figures, dates, durations, percentages.
6. If not in context, say exactly:
   "The information is not available in the provided policy document."
7. Do not repeat the same point in different words.
8. Language must be professional and precise.

Example Output:
**Premium Payment in Instalments**
- Grace Period of 15 days to pay instalment premium (Clause 13.i).
  - No coverage during grace period (Clause 13.ii).
  - Continuity benefits maintained if paid within grace period (Clause 13.iii).
  - No interest charged for late instalments (Clause 13.iv).
  - Policy cancelled if unpaid after grace period (Clause 13.v).

**Renewal of Policy**
- Renewable within 30 days to maintain continuity (Clause 10.iv).
- No coverage during grace period (Clause 10.iv).

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": PROMPT}
        )

        print("[INFO] RAG chain created successfully")
        return chain

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create RAG chain: {str(e)}")

import inspect

async def process_questions_with_rag(questions: List[str], rag_chain) -> List[str]:
    """Process multiple questions using RAG chain (robust across langchain versions)."""
    answers = []
    total_questions = len(questions)

    print(f"[INFO] Processing {total_questions} questions")

    for i, question in enumerate(questions, 1):
        try:
            print(f"[INFO] Processing question {i}/{total_questions}: {question[:80]}...")
            start_time = time.time()
            result = None
            last_exception = None

            # --- 1) Try async invocation methods if available (prefer these) ---
            async_names = ["ainvoke", "acall", "arun", "arun_chain", "__acall__"]
            for name in async_names:
                if hasattr(rag_chain, name):
                    method = getattr(rag_chain, name)
                    if inspect.iscoroutinefunction(method):
                        try:
                            # try plain string first
                            result = await method(question)
                            break
                        except Exception as e:
                            last_exception = e
                            # try common dict forms
                            for form in ({"question": question}, {"query": question}):
                                try:
                                    result = await method(form)
                                    break
                                except Exception as ee:
                                    last_exception = ee
                            if result is not None:
                                break

            # --- 2) If not async-result, run multiple sync attempts in a background thread ---
            if result is None:
                def sync_attempts(q):
                    last_exc = None
                    attempts = [
                        lambda x: getattr(rag_chain, "invoke")(x) if hasattr(rag_chain, "invoke") else None,
                        lambda x: getattr(rag_chain, "invoke")({"question": x}) if hasattr(rag_chain, "invoke") else None,
                        lambda x: getattr(rag_chain, "invoke")({"query": x}) if hasattr(rag_chain, "invoke") else None,
                        lambda x: getattr(rag_chain, "run")(x) if hasattr(rag_chain, "run") else None,
                        lambda x: getattr(rag_chain, "run")({"question": x}) if hasattr(rag_chain, "run") else None,
                        lambda x: getattr(rag_chain, "run")({"query": x}) if hasattr(rag_chain, "run") else None,
                        # try direct call forms
                        lambda x: rag_chain(x) if callable(rag_chain) else None,
                        lambda x: rag_chain({"question": x}) if callable(rag_chain) else None,
                        lambda x: rag_chain({"query": x}) if callable(rag_chain) else None,
                    ]

                    for fn in attempts:
                        if fn is None:
                            continue
                        try:
                            res = fn(q)
                            # Some callables may return None or raise; accept any non-None
                            if res is not None:
                                return res
                        except Exception as e:
                            last_exc = e
                            continue
                    # if nothing succeeded, re-raise the last exception for outer handler
                    if last_exc:
                        raise last_exc
                    return None

                try:
                    result = await asyncio.to_thread(sync_attempts, question)
                except Exception as e:
                    last_exception = e
                    result = None

            # --- 3) Normalize the result into a single string answer ---
            if isinstance(result, dict):
                # prefer common keys
                answer = (
                    result.get("result")
                    or result.get("answer")
                    or result.get("output_text")
                    or result.get("text")
                    or (next(iter(result.values())) if len(result) else "")
                )
            else:
                answer = "" if result is None else str(result)

            answer = answer.strip() if isinstance(answer, str) else str(answer).strip()

            if not answer:
                print(f"[WARNING] No answer produced for question {i}. Last exception: {last_exception}")
                answers.append("No answer generated.")
            else:
                print(f"[DEBUG] LLM Output (question {i}): {answer[:200]}")
                answers.append(answer)

            print(f"[INFO] Question {i} processed in {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"[ERROR] Error processing question {i}: {str(e)}")
            # keep going; return a per-question failure message (so indices align)
            answers.append("Unable to process this question due to an error.")

    return answers



# ================= API Endpoints =================

@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(request: HackRXRequest, token: str = Depends(verify_token)):
    """Main endpoint for HackRX evaluation system"""
    start_time = time.time()
    temp_file_path = None
    
    try:
        print(f"[INFO] Received request with {len(request.questions)} questions")
        print(f"[INFO] Document URL: {request.documents}")
        
        temp_file_path = await download_pdf_from_url(request.documents)
        documents = load_and_process_pdf(temp_file_path)
        vectorstore = create_vectorstore(documents)
        rag_chain = create_rag_chain(vectorstore)
        answers = await process_questions_with_rag(request.questions, rag_chain)

        for i, (q, a) in enumerate(zip(request.questions, answers), 1):
            print(f"[RESULT] Q{i}: {q}")
            print(f"[RESULT] A{i}: {a}")
        
        if len(answers) != len(request.questions):
            raise HTTPException(
                status_code=500,
                detail=f"Mismatch in questions ({len(request.questions)}) and answers ({len(answers)})"
            )
        
        total_time = time.time() - start_time
        print(f"[INFO] Successfully processed all questions in {total_time:.2f}s")
        
        return HackRXResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error in hackrx_run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"[INFO] Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                print(f"[WARNING] Failed to clean up temp file: {str(e)}")

@app.get("/policy.pdf")
async def serve_policy_pdf():
    """Serve local policy.pdf file for testing"""
    pdf_path = "policy.pdf"
    
    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=404,
            detail="policy.pdf not found. Please place policy.pdf in the same directory as app.py"
        )
    
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename="policy.pdf"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "HackRX RAG System is running",
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HackRX RAG System API",
        "version": "1.0.0",
        "description": "LLM-Powered Document Query System for Insurance Policies",
        "endpoints": {
            "main": "/hackrx/run",
            "health": "/health",
            "docs": "/docs"
        }
    }

# ================= Testing Functions =================

async def test_with_local_pdf(pdf_path: str, questions: List[str]):
    """Test the system with a local PDF file"""
    try:
        print(f"[TEST] Testing with local PDF: {pdf_path}")
        print(f"[TEST] Questions: {len(questions)}")
        
        documents = load_and_process_pdf(pdf_path)
        vectorstore = create_vectorstore(documents)
        rag_chain = create_rag_chain(vectorstore)
        answers = await process_questions_with_rag(questions, rag_chain)
        
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        
        for i, (question, answer) in enumerate(zip(questions, answers), 1):
            print(f"\nQ{i}: {question}")
            print(f"A{i}: {answer}")
            print("-" * 80)
        
        response = {"answers": answers}
        print(f"\nAPI Response Format:")
        print(json.dumps(response, indent=2))
        
        return response
        
    except Exception as e:
        print(f"[ERROR] Test failed: {str(e)}")
        return None

async def test_with_url(url: str, questions: List[str]):
    """Test the system with a PDF URL"""
    try:
        print(f"[TEST] Testing with URL: {url}")
        
        request = HackRXRequest(documents=url, questions=questions)
        temp_file_path = await download_pdf_from_url(request.documents)
        documents = load_and_process_pdf(temp_file_path)
        vectorstore = create_vectorstore(documents)
        rag_chain = create_rag_chain(vectorstore)
        answers = await process_questions_with_rag(request.questions, rag_chain)
        
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        print("\n" + "="*80)
        print("URL TEST RESULTS")
        print("="*80)
        
        for i, (question, answer) in enumerate(zip(questions, answers), 1):
            print(f"\nQ{i}: {question}")
            print(f"A{i}: {answer}")
            print("-" * 80)
        
        response = {"answers": answers}
        print(f"\nAPI Response Format:")
        print(json.dumps(response, indent=2))
        
        return response
        
    except Exception as e:
        print(f"[ERROR] URL test failed: {str(e)}")
        return None

# ================= Main Execution =================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HackRX RAG System")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--test-local", type=str, help="Test with local PDF file")
    parser.add_argument("--test-url", type=str, help="Test with PDF URL")
    parser.add_argument("--questions", nargs="+", help="Questions for testing")
    parser.add_argument("--port", type=int, default=8000, help="Port for server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for server")
    
    args = parser.parse_args()
    
    if args.serve:
        print(f"[INFO] Starting HackRX RAG System on {args.host}:{args.port}")
        print(f"[INFO] API Documentation: http://{args.host}:{args.port}/docs")
        print(f"[INFO] Health Check: http://{args.host}:{args.port}/health")
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.test_url:
        if not args.questions:
            print("[ERROR] Provide --questions for testing")
            exit(1)
        
        # If testing localhost URL, start server in background
        if "localhost:8000" in args.test_url or "127.0.0.1:8000" in args.test_url:
            import threading
            import uvicorn
            
            def run_server():
                uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
            
            print("[INFO] Starting background server for localhost testing...")
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            time.sleep(3)
        
        asyncio.run(test_with_url(args.test_url, args.questions))    
    
    elif args.test_local:
        if not args.questions:
            print("[ERROR] Provide --questions for testing")
            exit(1)
        asyncio.run(test_with_local_pdf(args.test_local, args.questions))
    
    else:
        print("HackRX RAG System")
        print("================")
        print()
        print("Usage:")
        print("  --serve                    Start the API server")
        print("  --test-local <pdf_path>    Test with local PDF")
        print("  --test-url <url>           Test with PDF URL")
        print("  --questions <q1> <q2>      Questions for testing")
        print()
        print("Examples:")
        print("  python rag.py --serve --port 8000")
        print("  python rag.py --test-local policy.pdf --questions 'What is grace period?'")
        print("  python rag.py --test-url 'http://localhost:8000/policy.pdf' --questions 'Coverage details?'")
