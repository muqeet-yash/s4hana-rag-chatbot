from flask import Flask, request, jsonify
from ctransformers import AutoModelForCausalLM
import os
import json
import numpy as np
from datetime import datetime
import warnings
from hdbcli import dbapi
import requests
warnings.filterwarnings("ignore")

app = Flask(__name__)
port = int(os.environ.get('PORT', 8080))

# ============ LOAD LLM ============
MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
print(f"📚 Loading TinyLlama from {MODEL_PATH}...")
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    model_type="llama",
    max_new_tokens=128,
    temperature=0.3,
    threads=2
)
print("✅ TinyLlama loaded")

# ============ LIGHTWEIGHT EMBEDDING (No PyTorch!) ============
def get_embedding(text):
    """Use a simple TF-IDF or random embedding to avoid PyTorch"""
    # Simple hash-based embedding (no ML libraries needed)
    words = text.lower().split()
    emb = np.zeros(384)
    for i, word in enumerate(words[:100]):  # Limit words
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

def search_similar(query, top_k=3):
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

def generate_answer(question, context):
    prompt = f"""<|system|>
Answer based ONLY on context. If no info, say "I don't have information about that."

<|user|>
Context: {context[:1000]}

Question: {question}

Answer:
<|assistant|>"""
    
    response = llm(prompt, max_new_tokens=128, temperature=0.3)
    return response

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "llm": "TinyLlama-1.1B",
        "embedding": "hash-based (lightweight)",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        similar_docs = search_similar(question)
        
        if not similar_docs:
            answer = "No relevant documents found."
        else:
            context = "\n\n".join(similar_docs)
            answer = generate_answer(question, context)
        
        return jsonify({"answer": answer})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("RAG Chatbot with TinyLlama (Lightweight)")
    print("=" * 50)
    print(f"🚀 Starting server on port {port}")
    print(f"💬 Ask: POST http://localhost:{port}/ask")
    print("=" * 50)
    app.run(host='0.0.0.0', port=port, debug=False)