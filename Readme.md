# RAG application with Groq API

It uses:
1. llama3-8b-8192 as LLM 
2. Huggingface embeddings
3. FAISS for vector embeddings
4. PyPDF for PDF document loading
5. RecursiveCharacterTextSplitter for text splitting
6. Streamlit for the web interface

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```
2. Set up the environment variables:
```bash
export GROQ_API_KEY=your_groq_api_key
```
3. Run the application:
```bash
streamlit run app.py
```
