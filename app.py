import streamlit as st
import os
from dotenv import load_dotenv

# Must be the first Streamlit command
st.set_page_config(
    page_title="DocuRAG-Search",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Import our backend modules
from src.ingestion import process_uploaded_pdf
from src.embeddings import create_vector_store, load_vector_store
from src.chain import create_rag_chain, ask_question

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
        
def process_pdf(uploaded_file):
    """Handle PDF upload, chunking, and embedding."""
    try:
        with st.spinner("📄 Reading PDF and splitting into chunks..."):
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name
            
            # Process PDF to chunks
            chunks = process_uploaded_pdf(file_bytes, filename)
            st.toast(f"Created {len(chunks)} chunks from {filename}", icon="✅")
            
        with st.spinner("🧠 Generating embeddings and building FAISS vector store..."):
            # Create vector store
            vectorstore = create_vector_store(chunks)
            st.session_state.vectorstore = vectorstore
            st.session_state.current_file = filename
            
            # Initialize RAG chain
            st.session_state.rag_chain = create_rag_chain(vectorstore)
            
        st.success(f"Ready! You can now ask questions about '{filename}'.")
        
        # Clear chat history for new file
        st.session_state.messages = []
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

def main():
    initialize_session_state()
    
    # Check for API Key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        st.error("⚠️ **GROQ_API_KEY is missing!** Please add your Groq API key to the `.env` file to enable the chat.")
    
    # Sidebar
    with st.sidebar:
        st.title("📄 DocuRAG-Search")
        st.markdown("Upload a PDF and ask questions about its content.")
        
        st.divider()
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload your PDF", 
            type=["pdf"],
            help="Limit 50MB per file"
        )
        
        if uploaded_file and uploaded_file.name != st.session_state.current_file:
            process_pdf(uploaded_file)
            
        # Try to load existing vectorstore on startup if none is loaded
        if not st.session_state.vectorstore and not uploaded_file:
            with st.spinner("Checking for existing database..."):
                vs = load_vector_store()
                if vs:
                    st.session_state.vectorstore = vs
                    st.session_state.rag_chain = create_rag_chain(vs)
                    st.session_state.current_file = "Loaded from disk"
                    st.success("Loaded existing vector database.")
                    
        st.divider()
        st.markdown("### About")
        st.markdown("""
        **Tech Stack:**
        - 🦜🔗 LangChain
        - 🧠 FAISS
        - 🤗 HuggingFace
        - ⚡ Groq (Llama 3)
        - 👑 Streamlit
        """)
        
    # Main Chat Area
    st.title("💬 Chat with your PDF")
    
    # Show active file context
    if st.session_state.current_file:
        st.info(f"Currently chatting with: **{st.session_state.current_file}**")
    else:
        st.info("👈 Please upload a PDF in the sidebar to get started.")
        
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}** (from {source.metadata.get('source', 'unknown')}):")
                        st.markdown(f"> {source.page_content}")
                        st.divider()

    # Chat Input
    if prompt := st.chat_input("Ask a question about the document..."):
        if not st.session_state.rag_chain:
            st.warning("Please upload a PDF first!")
            return
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                try:
                    # Query the RAG chain
                    result = ask_question(st.session_state.rag_chain, prompt)
                    
                    answer = result["answer"]
                    sources = result["sources"]
                    
                    message_placeholder.markdown(answer)
                    
                    # Add sources in an expander
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i+1}** (from {source.metadata.get('source', 'unknown')}):")
                            st.markdown(f"> {source.page_content}")
                            st.divider()
                            
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()
