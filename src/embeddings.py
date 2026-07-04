import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Define the persistent directory for the vector store
VECTORSTORE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vectorstore")

def get_embeddings_model():
    """
    Initialize and return the HuggingFace embeddings model.
    Using all-MiniLM-L6-v2 as it's fast, free, and runs locally.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(chunks: List[Document]):
    """
    Create a FAISS vector store from document chunks and save it to disk.
    """
    embeddings = get_embeddings_model()
    
    # Create the vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Ensure the directory exists
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    
    # Save it locally
    vectorstore.save_local(VECTORSTORE_DIR)
    
    return vectorstore

def load_vector_store():
    """
    Load an existing FAISS vector store from disk.
    Returns None if the store doesn't exist yet.
    """
    if not os.path.exists(os.path.join(VECTORSTORE_DIR, "index.faiss")):
        return None
        
    embeddings = get_embeddings_model()
    
    # Load the vector store with dangerous deserialization explicitly allowed 
    # since we created the file locally
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    return vectorstore

if __name__ == "__main__":
    # Smoke test
    print("Testing embeddings and vector store...")
    from ingestion import load_pdf, split_documents
    
    sample_pdf = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample.pdf")
    
    if os.path.exists(sample_pdf):
        print("1. Loading and splitting PDF...")
        docs = load_pdf(sample_pdf)
        chunks = split_documents(docs)
        
        print("2. Creating and saving vector store...")
        vs = create_vector_store(chunks)
        print("   Vector store saved successfully.")
        
        print("3. Loading vector store from disk...")
        loaded_vs = load_vector_store()
        
        if loaded_vs:
            print("   Vector store loaded successfully.")
            
            print("4. Testing similarity search...")
            query = "What is the main topic?"
            results = loaded_vs.similarity_search(query, k=1)
            
            if results:
                print(f"   Top match for '{query}':")
                print(f"   {results[0].page_content[:150]}...")
                
            print("\n✅ Embeddings pipeline test passed!")
        else:
            print("❌ Failed to load vector store.")
    else:
        print("⚠️ Skipping test: No sample PDF found.")
