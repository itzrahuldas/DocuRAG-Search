import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

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
        model_name="llama3-8b-8192", # Fast and capable model
        api_key=api_key
    )

def create_rag_chain(vectorstore):
    """
    Create a Retrieval-Augmented Generation (RAG) chain.
    """
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Define the system prompt for the QA system
    system_prompt = (
        "You are a helpful and intelligent assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create the document chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the final retrieval chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def ask_question(rag_chain, question: str):
    """
    Ask a question using the RAG chain and return the answer and source documents.
    """
    response = rag_chain.invoke({"input": question})
    return {
        "answer": response["answer"],
        "sources": response["context"]
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
            chain = create_rag_chain(vs)
            
            question = "What is the candidate's name?"
            print(f"3. Asking question: {question}")
            
            result = ask_question(chain, question)
            print("\nAnswer:")
            print(result["answer"])
            
            print("\nSources:")
            for i, doc in enumerate(result["sources"]):
                print(f"- Source {i+1} from {doc.metadata.get('source', 'unknown')}")
                
            print("\n✅ RAG chain test passed!")
        else:
            print("❌ No vector store found. Run embeddings.py first.")
    except Exception as e:
        print(f"❌ Error: {e}")
