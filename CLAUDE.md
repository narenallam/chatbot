# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered document chatbot with RAG (Retrieval Augmented Generation) capabilities. The system allows users to upload documents (PDF, DOCX, TXT, etc.) and have intelligent conversations about their content using local AI models via Ollama.

### Architecture

**Full-Stack Application:**
- **Frontend**: React TypeScript application with styled-components for UI
- **Backend**: FastAPI Python server with document processing and AI services
- **AI Models**: Local Ollama integration (Llama3:8b-instruct-q8_0)
- **Database**: ChromaDB for vector embeddings, SQLite for metadata
- **Document Processing**: Advanced OCR support with parallel processing

## Common Development Commands

### Backend Development
```bash
# Start backend server (from root directory)
cd backend && source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Install backend dependencies
cd backend && pip install -r requirements.txt

# Run backend tests
cd backend && python scripts/run_comprehensive_tests.py

# Reset system data
cd backend && ./scripts/cleanup_system.sh
```

### Frontend Development
```bash
# Start React development server
cd frontend && npm start

# Install frontend dependencies
cd frontend && npm install

# Build frontend for production
cd frontend && npm run build

# Run frontend tests
cd frontend && npm test
```

### Full Application Startup
```bash
# One-command startup (recommended)
./start_react_app.sh

# Or manually start both services:
# Terminal 1: Backend
cd backend && source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend && npm start
```

### AI Model Setup
```bash
# Setup Ollama and models
./setup_ollama.sh

# Setup OCR capabilities
./setup_ocr.sh
```

## Architecture Details

### Backend Services Architecture

**Core Services Location: `backend/app/services/`**

1. **Document Service** (`document_service.py`)
   - Handles file upload, conversion, and text extraction
   - Supports PDF, DOCX, TXT, HTML, CSV, XLS, XLSX, PPT, PPTX, images
   - Auto-converts all files to PDF for consistent processing

2. **OCR Document Service** (`ocr_document_service.py`)
   - Advanced OCR processing for scanned documents
   - Parallel processing across CPU cores for large files
   - Tesseract OCR with 100+ language support

3. **Parallel PDF Service** (`parallel_pdf_service.py`)
   - True multiprocessing for large PDF files (>10MB)
   - Splits PDFs into chunks and processes across CPU cores
   - Memory-efficient handling of 500MB+ files

4. **Vector Service** (`vector_service.py`)
   - ChromaDB integration for semantic search
   - Embedding generation using sentence-transformers
   - Similarity search for RAG context retrieval

5. **Chat Service** (`chat_service.py`)
   - Conversation management and history
   - Streaming responses integration
   - RAG context assembly and prompt engineering

6. **Database Service** (`database_service.py`)
   - SQLite operations for metadata and chat history
   - Session management and document tracking

### API Routes (`backend/app/api/routes/`)

- **Chat** (`chat.py`): Real-time streaming chat with RAG context
- **Upload** (`upload.py`): Multi-file upload with progress tracking
- **Documents** (`documents.py`): Document management and metadata
- **Admin** (`admin.py`): System administration endpoints
- **Generate** (`generate.py`): Content generation utilities

### Frontend Component Architecture

**Location: `frontend/src/components/`**

1. **App.tsx**: Main application shell with three-panel layout
2. **ChatPanel.tsx**: Real-time chat interface with streaming responses
3. **FileUploadPanel.tsx**: Drag-and-drop file upload with progress tracking
4. **ConversationPanel.tsx**: Chat history and conversation management
5. **StatusPanel.tsx**: System status monitoring (backend, AI models, documents)
6. **Console.tsx**: Real-time processing logs via WebSocket
7. **DocumentPreview.tsx**: Document viewing and source references
8. **DocumentSources.tsx**: RAG context source display

### Key Technology Integration

**Real-time Features:**
- WebSocket connection for live processing logs (`/ws/logs`)
- Server-sent events for streaming chat responses (`/api/chat/stream`)
- Real-time status monitoring and CPU usage tracking

**Document Processing Pipeline:**
1. File upload → SHA256 hash calculation → duplicate detection
2. Auto-conversion to PDF → text extraction (OCR if needed)
3. Text chunking → embedding generation → ChromaDB storage
4. Metadata storage in SQLite

**RAG Implementation:**
1. User query → embedding generation
2. Similarity search in ChromaDB → top-K context retrieval
3. Context + conversation history → Ollama LLM
4. Streaming response with source references

## Configuration

### Environment Variables (.env in backend/)
```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3:8b-instruct-q8_0
OLLAMA_BASE_URL=http://localhost:11434
MAX_FILE_SIZE_MB=500
ENABLE_OCR=true
ENABLE_PARALLEL_PROCESSING=true
```

### Service URLs
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Ollama**: http://localhost:11434

## Development Notes

### File Processing
- All files are converted to PDF for consistent processing
- Large files (>10MB) automatically use parallel processing
- OCR is auto-detected for scanned documents
- SHA256 hashing prevents duplicate uploads

### Testing
- Comprehensive test suite in `backend/scripts/run_comprehensive_tests.py`
- Tests real file processing with 5MB, 19MB, and 107MB samples
- Performance benchmarking and parallel processing validation

### Performance Considerations
- Parallel processing utilizes min(cpu_count(), 8) cores
- Memory-efficient chunking for 500MB+ files
- Real-time progress tracking for large document processing
- WebSocket logging prevents UI blocking during processing

### Code Patterns
- **Backend**: FastAPI with async/await, Pydantic models, dependency injection
- **Frontend**: React hooks, styled-components, TypeScript interfaces
- **Error Handling**: Comprehensive try-catch with user-friendly messages
- **State Management**: React Context for global state, localStorage for persistence

### Important Files to Understand
- `backend/app/main.py`: FastAPI application setup and WebSocket manager
- `backend/app/core/config.py`: Centralized configuration management
- `frontend/src/App.tsx`: Main application state and component orchestration
- `backend/app/services/document_service.py`: Core document processing logic

This architecture supports enterprise-grade document processing with complete privacy (local AI), advanced OCR capabilities, and real-time user feedback.