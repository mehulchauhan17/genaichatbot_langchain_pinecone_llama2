from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import CTransformers
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import time

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

embeddings = download_hugging_face_embeddings()

# ==================== PINECONE SETUP - 100% PROXY-SAFE VERSION ====================
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "genaichatbot"

print("Connecting directly to Pinecone index (bypassing proxy-sensitive calls)...")
try:
    index = pc.Index(index_name)
    print(f"SUCCESS: Connected to Pinecone index '{index_name}'")
except Exception as e:
    print(f"ERROR: Could not connect to index '{index_name}'")
    print("   → Create it manually here: https://app.pinecone.io")
    print("   → Name: genaichatbot | Dimension: 384 | Metric: cosine")
    print("   → Then run this app again.")
    exit(1)

time.sleep(1)

# ==================== REST OF YOUR CODE (unchanged) ====================
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.1}
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print(f"User: {msg}")
    result = qa({"query": msg})
    print(f"Bot: {result['result']}")
    return result["result"]

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)