from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from ctransformers import AutoModelForCausalLM
import os
import json
import numpy as np
from datetime import datetime
import warnings
from hdbcli import dbapi
import hashlib
import secrets
from typing import Optional

warnings.filterwarnings("ignore")

app = FastAPI(title="RAG Chatbot with TinyLlama")
security = HTTPBasic()

# ============ AUTHENTICATION SETUP ============
# Store hashed credentials (in production, use environment variables or database)
USERS = {
    "YashAdmin": {
        "password_hash": "35f6e77d93a47dc20b10bb5195c135b025db5155b1a465751a800d9d6e38529a",
        "role": "admin"
    }
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash"""
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    """Validate username and password"""
    user = USERS.get(credentials.username)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    if not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return {"username": credentials.username, "role": user["role"]}

# Optional: Add rate limiting for authentication attempts
failed_attempts = {}
def check_rate_limit(username: str):
    """Basic rate limiting to prevent brute force"""
    from time import time
    now = time()
    if username in failed_attempts:
        attempts, last_attempt = failed_attempts[username]
        if now - last_attempt < 300:  # 5 minutes window
            if attempts >= 5:  # Max 5 attempts
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many failed attempts. Try again later."
                )
            failed_attempts[username] = (attempts + 1, now)
        else:
            failed_attempts[username] = (1, now)
    else:
        failed_attempts[username] = (1, now)

# ============ LOAD LLM ============
MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
# https://cas-bridge.xethub.hf.co/xet-bridge-us/6591d4d754f88261730df832/015c9bb0376d9c3c9dab434ecb3bd57961dce1921a5b1bf134c6f1b824c25c8d?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20260406%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260406T061425Z&X-Amz-Expires=3600&X-Amz-Signature=a37914697147c28b93cfb080f1af559d5290aac9c349ff0980f431ccdbf96aed&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=67481a3cd3a83946f77edc8f&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf%3B+filename%3D%22tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf%22%3B&x-amz-checksum-mode=ENABLED&x-id=GetObject&Expires=1775459665&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3NTQ1OTY2NX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82NTkxZDRkNzU0Zjg4MjYxNzMwZGY4MzIvMDE1YzliYjAzNzZkOWMzYzlkYWI0MzRlY2IzYmQ1Nzk2MWRjZTE5MjFhNWIxYmYxMzRjNmYxYjgyNGMyNWM4ZCoifV19&Signature=IaEI2ToL2Z6mIWyWSrUElJH113laQJzCTvzetZTaEqz2d56GzK%7EHdi1Ax9oCmwpH9emROqNt2LINMKvM9U-fNtw5q62bNcGlDe%7ECHf1bZx9cLU7Jh4IeniT5v-G7KbxF3yH-99zf4HPNuLhzB5JgxvFYF0g2oluwiW1yST%7Ej9Gq3yQubJurXLpZ5yiBtYNJPvyFjFdsaQLwC3pPZXEIHtNGQ-qj5Zl3QV29AFRvaLp0hg3i-EAA-2ehIBg3TD7a0Ql7NTXiLMkjThldtFsaNn31fjl6OEjgDbQ8DszJ%7Euny37q1eANyItqLrSqZ0Ul6Am6z74Y8tE86N-vvIMcjoCw__&Key-Pair-Id=K2L8F4GPSG1IFC
print(f"📚 Loading TinyLlama from {MODEL_PATH}...")
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    model_type="llama",
    max_new_tokens=128,
    temperature=0.3,
    threads=2
)
print("✅ TinyLlama loaded")

# ============ LIGHTWEIGHT EMBEDDING ============
def get_embedding(text: str):
    """Simple hash-based embedding (no ML libraries needed)"""
    words = text.lower().split()
    emb = np.zeros(384)
    for i, word in enumerate(words[:100]):
        h = hash(word) % 384
        emb[h] += 1
    return emb / np.linalg.norm(emb)

# ============ HANA CONNECTION ============
def get_hana_connection():
    with open("SharedDevKey.json", "r") as f:
        service_key = json.load(f)
    conn = dbapi.connect(
        address=service_key["host"],
        port=int(service_key["port"]),
        user=service_key["user"],
        password=service_key["password"]
    )
    return conn, service_key["schema"]

def search_similar(query: str, top_k: int = 3):
    """Search using simple embeddings"""
    query_emb = get_embedding(query)
    conn, schema = get_hana_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(f'SELECT CONTENT, EMBEDDINGS FROM "{schema}"."COM_S4HANA_RAG_DOCUMENTCHUNKS"')
        rows = cursor.fetchall()
        
        if not rows:
            return []
        
        similarities = []
        for content, emb_str in rows:
            if emb_str:
                doc_emb = np.array([float(x) for x in emb_str.split(',')])
                similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
                similarities.append((similarity, content))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [content for _, content in similarities[:top_k]]
        
    except Exception as e:
        print(f"Search error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# def generate_answer(question: str, context: str):

#     prompt = f"""<|system|>
#         Your are an AI assistant. Answer based ONLY on context. 
#         If the context does not contain information to answer the question, say "I don't have information about that."

#         <|user|>
#         Context: {context[:1000]}

#         Question: {question}

#         Answer:
#         <|assistant|>"""
    
#     response = llm(prompt, max_new_tokens=1000, temperature=0.7)
#     return response

def generate_answer(question: str, context: str):
    # Clean and truncate context
    # context = context.strip()
    # if len(context) > 2000:
    #     context = context[:2000] + "..."  # Slightly larger for better comprehension
    
    prompt = f"""<|system|>
    You are a precise, fact-based AI assistant. Follow these rules strictly:

    1. Answer ONLY using information explicitly stated in the provided context
    2. If the context lacks sufficient information, say: "I don't have information about that in the provided context."
    3. Do not add external knowledge, assumptions, or general facts
    4. Keep answers concise, clear, and directly relevant to the question
    5. When quoting or referencing specific details, be faithful to the original wording
    6. If the context contains contradictory information, acknowledge the contradiction

    <|user|>
    ### Context:
    {context}

    ### Question:
    {question}

    ### Instructions:
    - Read the question carefully
    - Scan the context for relevant information
    - If found, answer directly using only context
    - If not found, state inability to answer

    ### Answer:
    <|assistant|>"""

    # Better LLM settings
    response = llm(
        prompt, 
        max_new_tokens=256,
        temperature=0.3,         # Zero temperature = deterministic, best for factual QA
        repetition_penalty=1.1,  # Discourage repetition,
        stop=["<|user|>", "</s>"]
    )
    
    # Optional: post-process to trim common prefixes
    response = response.strip()
    if response.lower().startswith(("answer:", "the answer is")):
        response = response.split(":", 1)[-1].strip()
    
    return response

# ============ REQUEST MODELS ============
class AskRequest(BaseModel):
    question: str

# ============ PUBLIC ROUTES (No Auth) ============
@app.get("/health")
def health():
    """Health check endpoint - no authentication required"""
    return {
        "status": "healthy",
        "llm": "TinyLlama-1.1B",
        "embedding": "hash-based (lightweight)",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/auth/status")
def auth_status():
    """Check if authentication is enabled"""
    return {"auth_enabled": True, "type": "Basic Authentication"}

# ============ PROTECTED ROUTES (Auth Required) ============
@app.post("/ask")
def ask(request: AskRequest, current_user: dict = Depends(get_current_user)):
    """Ask question - requires authentication"""
    try:
        question = request.question
        
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        
        similar_docs = search_similar(question)
        
        if not similar_docs:
            answer = "No relevant documents found."
        else:
            context = "\n\n".join(similar_docs)
            answer = generate_answer(question, context)
        
        return {
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/protected/test")
def test_auth(current_user: dict = Depends(get_current_user)):
    """Test endpoint to verify authentication is working"""
    return {
        "message": "Authentication successful!",
        "user": current_user["username"],
        "role": current_user["role"]
    }

# ============ ENTRY POINT ============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print("=" * 50)
    print("RAG Chatbot with TinyLlama (FastAPI)")
    print("=" * 50)
    print(f"🚀 Starting server on port {port}")
    print(f"💬 Ask: POST http://localhost:{port}/ask")
    print("=" * 50)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)