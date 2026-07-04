# 📄 DocuRAG-Search

> **Ask questions to your PDFs** — Powered by RAG (Retrieval-Augmented Generation)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)

---

## 🚀 What is DocuRAG-Search?

DocuRAG-Search is an intelligent document Q&A system that lets you upload PDFs and ask natural language questions about their content. It uses:

- **LangChain** for orchestrating the RAG pipeline
- **FAISS** for fast vector similarity search
- **HuggingFace Embeddings** (local, free) for text-to-vector conversion
- **Groq** (free API) for blazing-fast LLM inference
- **Streamlit** for a beautiful, interactive UI

## ✨ Features

- 📤 **PDF Upload** — Drag & drop or browse to upload PDF documents
- 🔍 **Smart Chunking** — Automatically splits documents for optimal retrieval
- ⚡ **Fast Search** — FAISS-powered vector similarity search
- 💬 **Chat Interface** — Conversational Q&A with your documents
- 📑 **Source References** — See exactly which pages answered your question
- 💾 **Persistent Storage** — Vector store persists between sessions

## 🛠️ Quick Start

### Prerequisites
- Python 3.10+
- Free [Groq API Key](https://console.groq.com)

### Installation

```bash
# Clone the repository
git clone https://github.com/itzrahuldas/DocuRAG-Search.git
cd DocuRAG-Search

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
copy .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser 🎉

## 📁 Project Structure

```
DocuRAG-Search/
├── src/
│   ├── ingestion.py      # PDF loading & text chunking
│   ├── embeddings.py     # HuggingFace embeddings & FAISS indexing
│   └── chain.py          # LangChain RAG chain with Groq LLM
├── app.py                # Streamlit UI
├── requirements.txt      # Dependencies
└── .env.example          # Environment template
```

## 🧠 How It Works

```
PDF Upload → Text Extraction → Chunking → Embedding → FAISS Index
                                                          ↓
User Question → Query Embedding → Similarity Search → Context + LLM → Answer
```

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built with ❤️ by [Rahul Das](https://github.com/itzrahuldas)**
