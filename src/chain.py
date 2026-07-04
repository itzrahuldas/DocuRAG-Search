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
        model_name="llama3-8b-8192",  # Fast and capable free model
        api_key=api_key
    )

def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(vectorstore):
    """
    Create a Retrieval-Augmented Generation (RAG) chain using LCEL.
    Returns a tuple: (chain, retriever) so we can also fetch sources.
    """
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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

    # LCEL chain: retrieve → format → prompt → llm → parse
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Return both the answer chain and the retriever so we can fetch sources separately
    return rag_chain, retriever

def ask_question(rag_chain_tuple, question: str) -> Dict[str, Any]:
    """
    Ask a question using the RAG chain and return the answer and source documents.
    """
    rag_chain, retriever = rag_chain_tuple

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
    from embeddings import load_vector_store

    try:
        vs = load_vector_store()
        if vs:
            print("1. Vector store loaded.")
            print("2. Creating RAG chain...")
            chain_tuple = create_rag_chain(vs)

            question = "What is the candidate's name?"
            print(f"3. Asking question: {question}")

            result = ask_question(chain_tuple, question)
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
