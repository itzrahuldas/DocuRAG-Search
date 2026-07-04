import os
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Model to use — check https://console.groq.com/docs/models for latest
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def get_llm():
    """
    Initialize the Groq LLM.
    Requires GROQ_API_KEY in the environment.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is not set or is invalid. Please check your .env file.")
        
    return ChatGroq(
        temperature=0,
        model_name=GROQ_MODEL,
        api_key=api_key
    )

def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(retriever):
    """
    Build a fresh RAG chain from a retriever using LCEL.
    Called every time we need a chain to avoid stale cached objects.
    """
    llm = get_llm()

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful and intelligent assistant for question-answering tasks. "
         "Use ONLY the following retrieved context to answer the question. "
         "If you don't know the answer from the context, say 'I don't have enough information in the document to answer that.' "
         "Keep the answer concise and to the point.\n\n"
         "Context:\n{context}"),
        ("human", "{question}"),
    ])

    # LCEL chain: retrieve -> format -> prompt -> llm -> parse
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def ask_question(vectorstore, question: str) -> Dict[str, Any]:
    """
    Ask a question by building a fresh chain each time.
    This avoids stale model/chain issues from Streamlit session caching.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Build a fresh chain every call — the LLM init is cheap
    rag_chain = build_rag_chain(retriever)

    # Get the answer
    answer = rag_chain.invoke(question)

    # Independently fetch source documents
    sources = retriever.invoke(question)

    return {
        "answer": answer,
        "sources": sources
    }

if __name__ == "__main__":
    # Smoke test
    print("Testing RAG Chain...")
    print(f"Using model: {GROQ_MODEL}")
    from embeddings import load_vector_store

    try:
        vs = load_vector_store()
        if vs:
            print("1. Vector store loaded.")

            question = "What is the candidate's name?"
            print(f"2. Asking question: {question}")

            result = ask_question(vs, question)
            print("\nAnswer:")
            print(result["answer"])

            print("\nSources:")
            for i, doc in enumerate(result["sources"]):
                print(f"- Source {i+1} from {doc.metadata.get('source', 'unknown')}")

            print("\n[SUCCESS] RAG chain test passed!")
        else:
            print("[ERROR] No vector store found. Run embeddings.py first.")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
