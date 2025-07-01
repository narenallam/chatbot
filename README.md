# 🤖 AI MATE - Intelligent Document Chatbot

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org) [![React](https://img.shields.io/badge/React-18.2-blue.svg)](https://reactjs.org) [![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-orange.svg)](https://ollama.ai)

---

## 🏗️ Architecture Diagram

```mermaid
flowchart TD
  classDef rounded fill:#232946,stroke:#a7c7e7,stroke-width:2px,color:#fff,rx:12,ry:12;
  classDef accent fill:#3b82f6,stroke:#fbbf24,stroke-width:2px,color:#fff,rx:12,ry:12;
  classDef ws fill:#10b981,stroke:#fbbf24,stroke-width:2px,color:#fff,rx:12,ry:12;
  classDef sse fill:#f59e42,stroke:#232946,stroke-width:2px,color:#fff,rx:12,ry:12;

  U1["User"]:::rounded
  F1["React App"]:::accent
  SSE["SSE (Server-Sent Events)"]:::sse
  WS["WebSocket<br/>Real-time"]:::ws
  B1["FastAPI App"]:::accent
  B2["Document Service"]:::rounded
  B3["Vector DB Service"]:::rounded
  B4["Multiprocessing Service"]:::rounded
  B5["Database Service"]:::rounded
  B6["RAG Logic"]:::rounded
  S1["File Storage"]:::rounded
  S2["Postgres DB"]:::rounded
  S3["Vector DB (FAISS/Chroma)"]:::rounded

  U1-->|Uploads/Queries|F1
  F1-->|REST API|B1
  F1-->|SSE|SSE
  F1-->|WebSocket|WS
  WS-->|Live Progress|B1
  SSE-->|Streaming Chat|B1
  B1-->|File Upload|B2
  B2-->|Save/Extract|S1
  B2-->|Text/Meta|B5
  B2-->|Parallel OCR/Extract|B4
  B4-->|Results|B2
  B2-->|Embeddings|B3
  B3-->|Vectors|S3
  B6-->|Query+RAG|B3
  B6-->|DB Lookup|B5
  B6-->|Results|F1
  B5-->|Meta|S2
  S2-->|Meta|B5
  S3-->|Vectors|B3
  S1-->|Files|B2

  %% Legends
  subgraph Legend [Legend]
    L1["Accent Service"]:::accent
    L2["Rounded Node"]:::rounded
    L3["WebSocket"]:::ws
    L4["SSE"]:::sse
  end
```

---

## 🌟 Features

- Advanced document processing (PDF, DOCX, PPTX, XLSX, PNG, JPG, HEIC, code/text/markdown)
- OCR and parallel processing for large/scanned files
- Duplicate prevention, real-time progress, and enhanced metadata
- RAG (Retrieval Augmented Generation) with local AI (Ollama)
- Modern, responsive UI/UX with live status and preview
- Privacy-first: all processing is local, no external APIs

---

## 📄 Flow Diagrams

### Real-Time Upload Progress
```mermaid
flowchart TD
  classDef rounded fill:#232946,stroke:#a7c7e7,stroke-width:2px,color:#fff,rx:12,ry:12;
  classDef ws fill:#10b981,stroke:#fbbf24,stroke-width:2px,color:#fff,rx:12,ry:12;
  A["User Drops Files"]:::rounded --> B["Frontend: Progress UI"]:::rounded
  B --> C["WebSocket Connection"]:::ws
  C --> D["Backend: File Analysis"]:::rounded
  D --> E["Progress: 10% - Analyzing..."]:::rounded
  E --> F["Text Extraction"]:::rounded
  F --> G["Progress: 60% - Extracting..."]:::rounded
  G --> H["Chunking & Embeddings"]:::rounded
  H --> I["Progress: 90% - Processing..."]:::rounded
  I --> J["Complete: 100% - Ready!"]:::rounded
```

### RAG Chat Flow
```mermaid
flowchart TD
  classDef rounded fill:#232946,stroke:#a7c7e7,stroke-width:2px,color:#fff,rx:12,ry:12;
  classDef accent fill:#3b82f6,stroke:#fbbf24,stroke-width:2px,color:#fff,rx:12,ry:12;
  classDef ws fill:#10b981,stroke:#fbbf24,stroke-width:2px,color:#fff,rx:12,ry:12;
  classDef sse fill:#f59e42,stroke:#232946,stroke-width:2px,color:#fff,rx:12,ry:12;
  A["User Types Question"]:::rounded --> B["Frontend: /api/chat/stream"]:::accent
  B-->|SSE|SSE["SSE (Server-Sent Events)"]:::sse
  B-->|WebSocket|WS["WebSocket"]:::ws
  B --> C["Chat Service: Query Processing"]:::rounded
  C --> D["Query Embedding Generation"]:::rounded
  D --> E["ChromaDB Similarity Search"]:::rounded
  E --> F["Context Assembly"]:::rounded
  F --> G["Prompt Engineering"]:::rounded
  G --> H["Ollama LLM Processing"]:::rounded
  H --> I["Streaming Response Generation"]:::rounded
  I --> J["Source References Assembly"]:::rounded
  J --> K["Frontend: Streaming Display"]:::accent
```

---

## 🧠 Architecture Explanation

- **Layered, service-oriented backend** (FastAPI, async, multiprocessing)
- **Frontend**: React, real-time WebSocket progress, modern UI
- **Document Service**: Handles all file types, duplicate detection, conversion, text extraction, chunking
- **Parallel Processing**: Large files and OCR handled in parallel using all CPU cores
- **RAG Logic**: Query embedding, vector search, context assembly, prompt engineering, LLM response, streaming output
- **Databases**: ChromaDB (vectors), SQLite (metadata), file storage (originals, converted)

---

## 📦 File Handling & Parallel Processing

- **Validation**: File type, size, and duplicate check (SHA256 hash)
- **Conversion**: Auto-PDF for office, PNG for HEIC, direct text for code/markdown
- **Extraction**: Text extraction with fallback to OCR for images/scanned PDFs
- **Parallelism**: Large files split and processed in parallel, with real-time progress
- **Stages**: Validation → Duplicate → Conversion → Extraction → Chunking → Embeddings → Storage

---

## 🧩 File Processing Stages & Embeddings

- **Stage 1**: Validation & duplicate detection
- **Stage 2**: Conversion (PDF, PNG, etc.)
- **Stage 3**: Text extraction (direct or OCR)
- **Stage 4**: Chunking (semantic splitting)
- **Stage 5**: Embedding generation (ChromaDB)
- **Stage 6**: Metadata and file storage

---

## 🔍 How RAG Works

- **User Query** → Embedded (vector)
- **Vector Search** → Top-K relevant chunks from ChromaDB
- **Context Assembly** → Relevant text + metadata
- **Prompt Engineering** → RAG context + user query
- **LLM (Ollama)** → Generates answer, streams to frontend
- **Source Attribution** → Shows which docs were used

---

## 🗄️ Databases

- **ChromaDB**: Vector storage for embeddings
- **SQLite**: Metadata for files, conversations, sessions
- **File Storage**: Originals, converted files, and extracted text

---

## 🛠️ REST API Endpoints

- `POST /api/upload` — Upload a document
- `GET /api/documents` — List all uploaded documents
- `GET /api/documents/{id}` — Get document details
- `DELETE /api/documents/{id}` — Delete a document
- `POST /api/chat/stream` — Start a chat with RAG
- `GET /api/status` — System status (backend, AI models)
- `POST /api/admin/reset` — Reset/clean system
- (See backend code for full OpenAPI schema)

---

## 🧪 Test Scripts

- **Comprehensive test suite**: `backend/tests/test_comprehensive_system.py`
- **Test runner**: `backend/tests/run_tests.sh`
- **Test data**: `test_data/` folder with real files
- **Performance metrics**: Throughput, memory, parallelism
- **How to run**:
  ```bash
  cd backend
  ./tests/run_tests.sh
  # or
  python tests/test_comprehensive_system.py
  ```
- **View results**:
  ```bash
  python scripts/display_test_report.py
  ```

---

## 🧹 Reset/Cleaning Scripts

- **cleanup_system.sh**: Safely wipes all data except .env and venv
- **reset_system_complete.sh**: Legacy, full system reset
- **display_test_report.py**: Beautiful test report viewer
- **How to use**:
  ```bash
  cd backend
  ./scripts/cleanup_system.sh --force
  # or
  ./scripts/reset_system_complete.sh
  ```

---

## 🚀 Project Setup

### Prerequisites
- Python 3.13+
- Node.js 18+
- Ollama (for local AI)

### 1. Setup Ollama & AI Models
```bash
brew install ollama
ollama serve
ollama pull llama3:8b-instruct-q8_0
```

### 2. Backend Setup
```bash
git clone <repository-url>
cd chatbot/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
npm start
```

---