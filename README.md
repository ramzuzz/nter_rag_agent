# NTER Hybrid RAG Agent 

An intelligent, strictly-scoped AI assistant designed for the **NTER construction company**. This agent uses a **Hybrid Retrieval-Augmented Generation (RAG)** approach to answer questions exclusively based on internal construction norms, safety rules, and SNiP documents.

## Key Features
- **Hybrid Search Engine**: Combines **ChromaDB** (for semantic vector search) and **BM25** (for exact keyword/code matching) using custom Reciprocal Rank Fusion.
- **Strictly Scoped Knowledge**: The AI is heavily prompted to avoid hallucinations and refuse answering off-topic questions (e.g., games, movies).
- **Graceful Fallbacks**: If the internal database lacks information, the AI politely informs the user rather than inventing facts.
- **Session Memory**: Uses unique Socket.IO session IDs to maintain chat context and remember previous questions.
- **Fast Startup Caching**: BM25 indexes and ChromaDB vectors are cached to disk (`.pkl` and `chroma_db/`), allowing for sub-second server restarts.
- **Markdown UI**: The frontend utilizes `marked.js` to render formatted, highly readable responses from the AI.

## Tech Stack
**AI & RAG:**
- Google GenAI SDK (Gemini 3.5 Flash-Lite)
- LangChain (Loaders, Text Splitters, Retrievers)
- HuggingFace Embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- ChromaDB (Vector Database)
- Rank-BM25 (Keyword Database)
- NLTK (Russian Snowball Stemmer)

**Backend:**
- Python 3.x
- FastAPI & Uvicorn (REST API)

**Frontend:**
- Node.js & Express
- Socket.IO (Real-time bi-directional communication)
- Vanilla HTML/JS + Marked.js

---

## How to install

### 1. Prerequisites
- Python 3.9+
- Node.js & npm
- A [Google Gemini API Key](https://ai.google.dev/)

### 2. Python Backend Setup
1. Clone the repository and navigate to the root folder.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install fastapi uvicorn google-genai langchain-community langchain-huggingface langchain-chroma pypdf sentence-transformers nltk python-dotenv rank_bm25
   ```
4. Create a `.env` file in the root directory and add your API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
5. Place your PDF documents (rules, norms, SNiPs) into the `docs/` folder.

### 3. Node.js Frontend Setup
1. Open a new terminal and navigate to the `web_client` folder:
   ```bash
   cd web_client
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```

---

## Running the Application

You need to run both the backend and the frontend simultaneously.

**Terminal 1 (Backend):**
```bash
# Ensure your virtual environment is active
uvicorn api:app --reload
```
*(On the first run, the app will read the PDFs and build the databases. Subsequent runs will load from cache instantly).*

**Terminal 2 (Frontend):**
```bash
cd web_client
node server.js
```

**Open your browser** and navigate to `http://localhost:3000` to chat with the agent!

---

## How the Hybrid RAG Works
1. **Document Ingestion**: PDFs from the `docs/` folder are split into 1000-character chunks with a 200-character overlap.
2. **Dual Indexing**:
   - Chunks are vectorized using HuggingFace and stored in ChromaDB.
   - Chunks are tokenized (with Russian stemming) and stored in a BM25 index.
3. **Querying**: When a user asks a question, the agent performs a similarity search in Chroma and a keyword search in BM25.
4. **Fusion**: A custom `reciprocal_rank_fusion` algorithm mathematically merges and ranks the top results from both searches.
5. **Generation**: The fused context is passed to Gemini, which acts as a strict corporate assistant to generate the final response.

---
*Note: There is an `.env` file and the generated databases (`chroma_db/`, `bm25_index.pkl`), so be careful.*
