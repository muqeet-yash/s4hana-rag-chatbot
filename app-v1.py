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
warnings.filterwarnings("ignore")

app = Flask(__name__)
port = int(os.environ.get('PORT', 8080))

# # ============ LOAD MODELS ============
# print("🚀 Loading models...")

# 1. Embedding model (for search)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
print(f"📚 Loading TinyLlama from {MODEL_PATH}...")
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    model_type="llama",
    max_new_tokens=256,
    temperature=0.3,
    threads=2
)
print("✅ TinyLlama loaded")

# ============ HANA CONNECTION ============


def search_similar(query, top_k=3):
    """Search for similar documents using cosine similarity"""
    # Create query embedding
    query_emb = embedder.encode([query])[0]
    
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
        # Get all chunks
        cursor.execute(f'SELECT CONTENT, EMBEDDINGS FROM "{schema}"."COM_S4HANA_RAG_DOCUMENTCHUNKS"')
        rows = cursor.fetchall()
        
        if not rows:
            return []
        
        # Calculate similarities
        similarities = []
        for content, emb_str in rows:
            if emb_str:
                doc_emb = np.array([float(x) for x in emb_str.split(',')])
                # Calculate cosine similarity
                similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
                similarities.append((similarity, content))
        
        # Sort by similarity and get top_k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [content for _, content in similarities[:top_k]]
        
    except Exception as e:
        print(f"Search error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def generate_answer(question, context):
    """Generate answer using TinyLlama"""
    prompt = f"""<|system|>
    You are a helpful assistant. Answer the question based ONLY on the context provided. 
    If the context does not contain information to answer the question, say "I don't have information about that in my documents."

    <|user|>
    Context: {context}

    Question: {question}

    Answer:

    <|assistant|>
    """
    
    response = llm(
        prompt,
        max_new_tokens=256,
        temperature=0.3,
        stop=["<|user|>", "</s>"]
    )
    
    return response


# ============ API ENDPOINTS ============
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models": {
            "embedding": "all-MiniLM-L6-v2",
            "llm": "TinyLlama-1.1B"
        },
        "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else "not found",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/ask', methods=['POST'])
def ask():
    """Ask a question (RAG pipeline)"""
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Search for relevant context
        similar_docs = search_similar(question)
        
        if not similar_docs:
            answer = "No relevant documents found. Please upload documents first."
        else:
            context = "\n\n".join(similar_docs)
            answer = generate_answer(question, context)
        
        # Just return answer - CAP will handle saving
        return jsonify({
            "answer": answer
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ MAIN ============
if __name__ == '__main__':
    print("=" * 50)
    print("RAG Chatbot with TinyLlama")
    print("=" * 50)
    
    
    print(f"🚀 Starting server on port {port}")
    print(f"📊 Health check: http://localhost:{port}/health")
    # print(f"📤 Upload: POST http://localhost:{port}/upload")
    print(f"💬 Ask: POST http://localhost:{port}/ask")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=port, debug=False)