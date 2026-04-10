from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM
import uuid
import os
import json
import numpy as np
from datetime import datetime
import warnings
from hdbcli import dbapi
from functools import lru_cache
import hashlib
warnings.filterwarnings("ignore")

app = Flask(__name__)
port = int(os.environ.get('PORT', 8080))

# ============ LOAD MODELS ============
embedder = SentenceTransformer('all-MiniLM-L6-v2')

MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
print(f"📚 Loading TinyLlama from {MODEL_PATH}...")
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    model_type="llama",
    max_new_tokens=128,  # Reduced from 256 for speed
    temperature=0.3,
    threads=4,  # Increased threads
    top_k=40,
    top_p=0.9
)
print("✅ TinyLlama loaded")

# ============ CACHE SETUP ============
# Cache for embeddings
@lru_cache(maxsize=100)
def get_embedding(text):
    return embedder.encode([text])[0]

# Cache for LLM responses
response_cache = {}
CACHE_SIZE = 50

def get_cached_response(question, context_hash):
    cache_key = f"{question}_{context_hash}"
    return response_cache.get(cache_key)

def set_cached_response(question, context_hash, answer):
    cache_key = f"{question}_{context_hash}"
    if len(response_cache) > CACHE_SIZE:
        # Remove oldest item
        response_cache.pop(next(iter(response_cache)))
    response_cache[cache_key] = answer

# ============ HANA CONNECTION ============
def get_chunks_from_db():
    """Fetch all chunks once and cache them"""
    with open("SharedDevKey.json", "r") as f:
        service_key = json.load(f)
    conn = dbapi.connect(
        address=service_key["host"],
        port=int(service_key["port"]),
        user=service_key["user"],
        password=service_key["password"]
    )
    schema = service_key["schema"]
    cursor = conn.cursor()
    
    try:
        cursor.execute(f'SELECT CONTENT, EMBEDDINGS FROM "{schema}"."COM_S4HANA_RAG_DOCUMENTCHUNKS"')
        rows = cursor.fetchall()
        return rows
    finally:
        cursor.close()
        conn.close()

# Cache chunks in memory
CHUNKS_CACHE = None
LAST_CACHE_TIME = None

def get_cached_chunks():
    global CHUNKS_CACHE, LAST_CACHE_TIME
    # Refresh cache every 5 minutes
    if CHUNKS_CACHE is None or (datetime.now() - LAST_CACHE_TIME).seconds > 300:
        CHUNKS_CACHE = get_chunks_from_db()
        LAST_CACHE_TIME = datetime.now()
    return CHUNKS_CACHE

def search_similar_fast(query, top_k=3):
    """Optimized search using cached data"""
    query_emb = get_embedding(query)
    rows = get_cached_chunks()
    
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

def generate_answer_fast(question, context):
    """Generate answer with caching"""
    context_hash = hashlib.md5(context.encode()).hexdigest()
    
    # Check cache
    cached = get_cached_response(question, context_hash)
    if cached:
        return cached
    
    prompt = f"""<|system|>Answer based ONLY on context. If no info, say "I don't have information about that."

    <|user|>
    Context: {context[:1000]}

    Question: {question}

    Answer:
    <|assistant|>"""
    
    response = llm(prompt, max_new_tokens=128, temperature=0.3)
    
    # Cache response
    set_cached_response(question, context_hash, response)
    return response



# ============ GENERAL ASK (NO RAG) ============
@app.route('/ask_general', methods=['POST'])
def ask_general():
    """Ask question without RAG - general knowledge only"""
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        prompt = f"""<|system|>You are a helpful assistant. Answer general questions concisely.

            <|user|>
            Question: {question}

            Answer:
            <|assistant|>"""
        
        response = llm(prompt, max_new_tokens=256, temperature=0.7)
        
        return jsonify({
            "answer": response,
            "mode": "general"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/identify_pattern', methods=['POST'])
def identify_pattern():
    """Identify patterns in the provided text"""
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        prompt = f"""
        <|system|>
        You are a helpful assistant. Analyze the provided text and identify any patterns, trends, or insights. 
        Be concise and focus on key observations. 

        Return the response strictly in JSON format with the following keys:
        - "intent": The main purpose or intent of the text.
        - "variables": A list of objects with "name", "type", and "value" (e.g., customer, product, date).
        - "insights": Key observations or trends. If none, return "No significant patterns identified."

        <|user|>
        Question: {question}

        <|assistant|>
        """
        
        response = llm(prompt, max_new_tokens=256, temperature=0.7)
        print(f"Raw response: {response}")
        return jsonify({
            "answer": response,
            "mode": "general"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ RAG ASK (WITH DOCUMENTS) ============
@app.route('/ask', methods=['POST'])
def ask():
    """Ask question with RAG - uses uploaded documents"""
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Search using cached chunks
        similar_docs = search_similar_fast(question)
        
        if not similar_docs:
            answer = "No relevant documents found. Use /ask_general for general questions."
        else:
            context = "\n\n".join(similar_docs)
            answer = generate_answer_fast(question, context)
        
        return jsonify({
            "answer": answer,
            "mode": "rag",
            "context_used": len(similar_docs)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ HEALTH CHECK ============
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "models": {
            "embedding": "all-MiniLM-L6-v2",
            "llm": "TinyLlama-1.1B"
        },
        "cache": {
            "chunks_loaded": len(get_cached_chunks()) if get_cached_chunks() else 0,
            "response_cache_size": len(response_cache)
        },
        "timestamp": datetime.now().isoformat()
    })

# ============ CLEAR CACHE ============
@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    global CHUNKS_CACHE, response_cache
    CHUNKS_CACHE = None
    response_cache.clear()
    get_embedding.cache_clear()
    return jsonify({"message": "Cache cleared"})

# ============ MAIN ============
if __name__ == '__main__':
    print("=" * 50)
    print("RAG Chatbot with TinyLlama (Optimized)")
    print("=" * 50)
    
    # Pre-load chunks
    print("📦 Pre-loading chunks from HANA...")
    get_cached_chunks()
    print(f"✅ Loaded {len(CHUNKS_CACHE) if CHUNKS_CACHE else 0} chunks")
    
    print(f"🚀 Starting server on port {port}")
    print(f"💬 RAG Ask: POST http://localhost:{port}/ask")
    print(f"💬 General Ask: POST http://localhost:{port}/ask_general")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)