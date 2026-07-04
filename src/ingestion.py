import os
import tempfile
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and return its pages as LangChain Documents.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
        
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split a list of LangChain Documents into smaller chunks for vector storage.
    Uses RecursiveCharacterTextSplitter for optimal semantic chunking.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

def process_uploaded_pdf(uploaded_file_bytes: bytes, filename: str) -> List[Document]:
    """
    Helper function to process a PDF uploaded via Streamlit.
    Saves the bytes to a temporary file, loads, and chunks it.
    """
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file_bytes)
        tmp_path = tmp_file.name
        
    try:
        # Load and split the documents
        documents = load_pdf(tmp_path)
        
        # Add metadata about the source filename
        for doc in documents:
            doc.metadata["source"] = filename
            
        chunks = split_documents(documents)
        return chunks
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    # Smoke test - requires a sample PDF in data/sample.pdf
    sample_pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample.pdf")
    
    print(f"Testing ingestion pipeline with: {sample_pdf_path}")
    
    if os.path.exists(sample_pdf_path):
        try:
            print("1. Loading PDF...")
            docs = load_pdf(sample_pdf_path)
            print(f"   Loaded {len(docs)} pages.")
            
            print("2. Splitting text...")
            chunks = split_documents(docs)
            print(f"   Created {len(chunks)} chunks.")
            
            if chunks:
                print("\nSample of first chunk:")
                print("-" * 50)
                # Safely encode to avoid Windows charmap errors
                print(chunks[0].page_content[:200].encode('ascii', 'replace').decode('ascii') + "...")
                print("-" * 50)
                print(f"Metadata: {chunks[0].metadata}")
                
            print("\n[SUCCESS] Ingestion pipeline test passed!")
        except Exception as e:
            print(f"[ERROR] Error during test: {e}")
    else:
        print(f"⚠️ Skipping test: No sample PDF found at {sample_pdf_path}")
