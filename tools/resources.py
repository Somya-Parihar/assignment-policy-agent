import json
import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
import faiss

load_dotenv()

# --- 1. LLM SETUP (Gemini) ---
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    google_api_key=api_key,
    model="gemini-2.5-pro", # Updated to a standard valid model name
    temperature=0 
)

# --- 2. MOCK TOOL ---
@tool
def calculate_premium(age: int, location: str, income: int):
    """Calculates the insurance premium based on user data."""
    base_rate = 100
    if age < 25: base_rate += 50
    if age >= 25: base_rate += 25
    
    return json.dumps({
        "status": "success",
        "quote_amount": base_rate,
        "currency": "INR",
        "user_data": {"age": age, "location": location, "income": income}
    })

# --- 3. RAG SETUP ---

PDF_PATH = "policy.pdf"
DB_PATH = "faiss_index_store"

# Initialize Embeddings (Must be same for saving and loading)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

def get_vector_store():
    """
    Logic to load existing index or create a new one from PDF.
    """
    # CASE 1: Load existing index
    if os.path.exists(DB_PATH):
        print(f"Loading existing FAISS index from {DB_PATH}...")
        return FAISS.load_local(
            DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )

    # CASE 2: Create new index from PDF
    print(f"Index not found. Creating new one from {PDF_PATH}...")
    
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: {PDF_PATH} not found. Returning empty store.")
        # Fallback empty store
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello")))
        return FAISS(embeddings, index, InMemoryDocstore(), {})

    # Load and Split
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(docs)
    
    # Create and Save
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(DB_PATH)
    print(f"Index saved to {DB_PATH}")
    
    return vector_store

# Initialize the retriever
vector_store = get_vector_store()
retriever = vector_store.as_retriever()