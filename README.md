# 🤖 AI MATE - Intelligent Document Chatbot

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18.2-blue.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-orange.svg)](https://ollama.ai)

A powerful AI-powered document chatbot with **Retrieval Augmented Generation (RAG)** capabilities. Upload documents and have intelligent conversations about their content using local AI models for complete privacy.

## 🌟 Features

### 📄 **Advanced Document Processing**
- **Supported Formats**: PDF, DOCX, PPTX, XLSX, PNG, JPG, HEIC
- **OCR Support**: Automatic text extraction from scanned documents and images
- **Duplicate Prevention**: SHA256 file hashing prevents duplicate uploads
- **Enhanced Metadata**: File size, upload time, processing strategy tracking
- **File Validation**: Type-specific validation and error handling
- **Real-Time Progress**: Live upload progress with percentage and stage tracking

### 🧠 **Intelligent RAG System**
- **Vector Search**: Semantic similarity search using ChromaDB
- **Context Retrieval**: Automatically finds relevant document chunks
- **Source References**: Shows which documents were used in responses
- **Streaming Responses**: Real-time chat with typewriter effect
- **Multi-Document Support**: Query across all uploaded documents
- **Smart File Processing**: Optimized parallel processing for better performance

### 🚀 **Modern UI/UX**
- **Three-Panel Layout**: File upload, chat interface, conversation history
- **Real-Time Progress Tracking**: WebSocket-based upload progress with live stages
- **Document Preview System**: Inline PDF and image preview with download options
- **Enhanced File Management**: Rich document details with file type icons
- **Console Logging**: Live processing logs with toggle visibility
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Clean Status Bar**: Simplified system status indicators

### 🔒 **Privacy-First Architecture**
- **Local AI Models**: Uses Ollama for complete data privacy
- **No External APIs**: All processing happens on your machine
- **Secure Storage**: Local SQLite and ChromaDB storage

### ⚡ **Performance Features**
- **Optimized Processing Strategy**: Enhanced parallel/sequential processing selection
- **Real-Time Feedback**: WebSocket-powered progress updates
- **Memory Optimization**: Efficient handling of large files (up to 500MB)
- **CPU Utilization**: Intelligent use of available CPU cores
- **Caching**: Duplicate detection and smart file management
- **Low-Latency Processing**: Faster response times with improved thresholds

## 🏗️ Architecture

### **Real-Time Upload Progress Flow**
```mermaid
graph TD
    A["User Drops Files"] --> B["Frontend: Initialize Progress UI"]
    B --> C["WebSocket Connection"]
    C --> D["Backend: File Analysis"]
    D --> E["Progress: 10% - Analyzing..."]
    E --> F["Text Extraction"]
    F --> G["Progress: 60% - Extracting..."]
    G --> H["Chunking & Embeddings"]
    H --> I["Progress: 90% - Processing..."]
    I --> J["Complete: 100% - Ready!"]
    
    style A fill:#2d3748,stroke:#4fd1c7,stroke-width:2px,color:#ffffff
    style B fill:#2d3748,stroke:#63b3ed,stroke-width:2px,color:#ffffff
    style C fill:#2d3748,stroke:#f6ad55,stroke-width:2px,color:#ffffff
    style D fill:#2d3748,stroke:#f6ad55,stroke-width:2px,color:#ffffff
    style E fill:#2d3748,stroke:#fbb6ce,stroke-width:2px,color:#ffffff
    style F fill:#2d3748,stroke:#f6ad55,stroke-width:2px,color:#ffffff
    style G fill:#2d3748,stroke:#fbb6ce,stroke-width:2px,color:#ffffff
    style H fill:#2d3748,stroke:#f6ad55,stroke-width:2px,color:#ffffff
    style I fill:#2d3748,stroke:#fbb6ce,stroke-width:2px,color:#ffffff
    style J fill:#2d3748,stroke:#68d391,stroke-width:2px,color:#ffffff
```

### **Document Preview System**
```mermaid
graph TD
    A["User Clicks Document Link"] --> B["Document Preview Modal Opens"]
    B --> C["Document Type Detection"]
    C --> D{"File Type?"}
    D -->|PDF| E["PDF Viewer (iframe)"]
    D -->|Image| F["Native Image Display"]
    D -->|Other| G["PDF Conversion Preview"]
    E --> H["Download Options"]
    F --> H
    G --> H
    H --> I["Original/PDF Download"]
    
    style A fill:#2d3748,stroke:#4fd1c7,stroke-width:2px,color:#ffffff
    style B fill:#2d3748,stroke:#63b3ed,stroke-width:2px,color:#ffffff
    style C fill:#2d3748,stroke:#f6ad55,stroke-width:2px,color:#ffffff
    style D fill:#2d3748,stroke:#fbb6ce,stroke-width:2px,color:#ffffff
    style E fill:#2d3748,stroke:#fc8181,stroke-width:2px,color:#ffffff
    style F fill:#2d3748,stroke:#fc8181,stroke-width:2px,color:#ffffff
    style G fill:#2d3748,stroke:#fc8181,stroke-width:2px,color:#ffffff
    style H fill:#2d3748,stroke:#9f7aea,stroke-width:2px,color:#ffffff
    style I fill:#2d3748,stroke:#68d391,stroke-width:2px,color:#ffffff
```

### **RAG Chat Flow**
```mermaid
graph TD
    A["User Types Question"] --> B["Frontend: /api/chat/stream"]
    B --> C["Chat Service: Query Processing"]
    C --> D["Query Embedding Generation<br/>(Vector Conversion)"]
    D --> E["ChromaDB Similarity Search<br/>(Top-K Document Retrieval)"]
    E --> F["Context Assembly<br/>(Relevant Chunks + Metadata)"]
    F --> G["Prompt Engineering<br/>(RAG Context + User Query)"]
    G --> H["Ollama LLM Processing<br/>(Local AI Generation)"]
    H --> I["Streaming Response Generation<br/>(Real-time Output)"]
    I --> J["Source References Assembly<br/>(Document Attribution)"]
    J --> K["Frontend: Streaming Display<br/>(Typewriter Effect + Sources)"]
    
    style A fill:#2d3748,stroke:#4fd1c7,stroke-width:2px,color:#ffffff
    style B fill:#2d3748,stroke:#63b3ed,stroke-width:2px,color:#ffffff
    style C fill:#2d3748,stroke:#63b3ed,stroke-width:2px,color:#ffffff
    style D fill:#2d3748,stroke:#f6ad55,stroke-width:2px,color:#ffffff
    style E fill:#2d3748,stroke:#fc8181,stroke-width:2px,color:#ffffff
    style F fill:#2d3748,stroke:#f6ad55,stroke-width:2px,color:#ffffff
    style G fill:#2d3748,stroke:#f6ad55,stroke-width:2px,color:#ffffff
    style H fill:#2d3748,stroke:#fbb6ce,stroke-width:2px,color:#ffffff
    style I fill:#2d3748,stroke:#9f7aea,stroke-width:2px,color:#ffffff
    style J fill:#2d3748,stroke:#9f7aea,stroke-width:2px,color:#ffffff
    style K fill:#2d3748,stroke:#68d391,stroke-width:2px,color:#ffffff
```

### **System Component Interaction**
```mermaid
graph LR
    UI["React Frontend<br/>Port 3000"] <--> API["FastAPI Backend<br/>Port 8000"]
    API <--> DB1["ChromaDB<br/>Vector Storage"]
    API <--> DB2["SQLite<br/>Metadata Storage"]
    API <--> AI["Ollama LLM<br/>Port 11434"]
    API <--> OCR["Tesseract OCR<br/>Image Processing"]
    UI <--> WS["WebSocket<br/>Real-time Progress"]
    
    subgraph "Document Processing"
        DOC["Document Service"]
        VEC["Vector Service"]
        CHAT["Chat Service"]
        PROG["Progress Tracking"]
    end
    
    API --> DOC
    API --> VEC
    API --> CHAT
    WS --> PROG
    
    style UI fill:#2d3748,stroke:#4fd1c7,stroke-width:2px,color:#ffffff
    style API fill:#2d3748,stroke:#63b3ed,stroke-width:2px,color:#ffffff
    style AI fill:#2d3748,stroke:#fbb6ce,stroke-width:2px,color:#ffffff
    style DB1 fill:#2d3748,stroke:#fc8181,stroke-width:2px,color:#ffffff
    style DB2 fill:#2d3748,stroke:#fc8181,stroke-width:2px,color:#ffffff
    style OCR fill:#2d3748,stroke:#f6ad55,stroke-width:2px,color:#ffffff
    style WS fill:#2d3748,stroke:#68d391,stroke-width:2px,color:#ffffff
    style DOC fill:#1a202c,stroke:#9f7aea,stroke-width:2px,color:#ffffff
    style VEC fill:#1a202c,stroke:#9f7aea,stroke-width:2px,color:#ffffff
    style CHAT fill:#1a202c,stroke:#9f7aea,stroke-width:2px,color:#ffffff
    style PROG fill:#1a202c,stroke:#68d391,stroke-width:2px,color:#ffffff
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+**
- **Node.js 18+**
- **Ollama** (for AI models)

### 1. Setup Ollama & AI Models

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
ollama serve

# Pull the required model
ollama pull llama3:8b-instruct-q8_0
```

### 2. Clone & Setup Backend

```bash
# Clone repository
git clone <repository-url>
cd chatbot

# Setup Python virtual environment
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Setup Frontend

```bash
# In a new terminal
cd frontend

# Install dependencies
npm install

# Start React development server
npm start
```

### 4. Access Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## ✨ New Features & Enhancements

### 🔄 **Real-Time Progress Tracking**
- **Live Upload Progress**: See file processing progress in real-time
- **Stage-by-Stage Updates**: Track each processing stage with percentages
- **WebSocket Integration**: Instant progress updates without polling
- **Batch Processing Stats**: Monitor multiple file uploads simultaneously
- **Error Handling**: Clear error messages with detailed feedback

### ⚡ **Enhanced Parallel Processing Strategy**
- **Optimized Thresholds**: Lower limits for better user experience
  - **Multiple Files**: Parallel processing for 2+ files when total > 5MB
  - **Large Files**: Files >10MB (reduced from 50MB) trigger parallel mode
  - **Batch Size**: 50MB+ total (reduced from 100MB) enables parallel processing
- **Smart CPU Utilization**: Automatic core allocation based on file complexity
- **Memory Efficiency**: Improved handling of large file batches
- **Database Optimization**: Resolved "readonly database" issues for seamless processing
- **Reduced Latency**: Faster processing start times with lower thresholds

### 📋 **Enhanced Document Management**
- **Inline Document Preview**: View PDFs and images directly in the app
- **Smart Download Options**: Download original files or PDF conversions
- **Document Type Icons**: Visual file type identification
- **Rich Metadata Display**: File size, upload date, processing time
- **Improved File Organization**: Better sorting and filtering options

### 🎨 **UI/UX Improvements**
- **Streamlined Status Bar**: Shows only essential system information
- **Responsive Progress Bars**: Animated progress with shimmer effects
- **Color-Coded Status**: Visual indicators for upload states
- **Mobile-Friendly Design**: Optimized for all screen sizes
- **Keyboard Shortcuts**: Quick navigation and actions

### ⚡ **Performance Optimizations**
- **Smart Processing Selection**: Automatic parallel/sequential strategy
- **Memory Management**: Efficient handling of large file batches
- **CPU Optimization**: Intelligent core utilization
- **Caching Improvements**: Faster duplicate detection and retrieval

## 📁 Project Structure

```
chatbot/
├── backend/                 # FastAPI Python backend
│   ├── app/
│   │   ├── api/routes/     # API endpoints
│   │   │   ├── chat.py     # RAG chat endpoints
│   │   │   ├── documents.py # Document management & preview
│   │   │   ├── upload.py   # File upload with progress tracking
│   │   │   └── admin.py    # System administration
│   │   ├── services/       # Core business logic
│   │   │   ├── chat_service.py      # RAG chat logic
│   │   │   ├── document_service.py  # Document processing
│   │   │   ├── vector_service.py    # ChromaDB operations
│   │   │   └── database_service.py  # SQLite operations
│   │   ├── core/
│   │   │   ├── config.py   # Configuration management
│   │   │   └── prompts.py  # AI prompt templates
│   │   └── main.py         # FastAPI application with WebSocket
│   ├── data/               # SQLite database & file storage
│   ├── embeddings/         # ChromaDB vector storage
│   └── requirements.txt    # Python dependencies
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── ChatPanel.tsx        # Chat interface
│   │   │   ├── FileUploadPanel.tsx  # Upload with progress tracking
│   │   │   ├── ConversationPanel.tsx # Chat history
│   │   │   ├── StatusPanel.tsx      # Simplified system status
│   │   │   ├── Console.tsx          # Processing logs
│   │   │   ├── DocumentDetails.tsx  # Document information
│   │   │   ├── DocumentPreview.tsx  # Enhanced document viewer
│   │   │   └── DocumentSources.tsx  # RAG source display
│   │   ├── styles/         # Styled components
│   │   └── App.tsx         # Main application
│   └── package.json        # Node.js dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Backend Configuration

Create `.env` file in the `backend/` directory:

```env
# AI Model Configuration
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3:8b-instruct-q8_0
OLLAMA_BASE_URL=http://localhost:11434

# Document Processing
MAX_FILE_SIZE_MB=500
ENABLE_OCR=true
ENABLE_PARALLEL_PROCESSING=true
PARALLEL_PROCESSING_THRESHOLD_MB=10  # Files >10MB use parallel processing
BATCH_PARALLEL_THRESHOLD_MB=5  # Multiple files >5MB total use parallel processing

# Database
DATABASE_URL=sqlite:///./data/chatbot.db
VECTOR_DB_PATH=./embeddings

# API Configuration
CORS_ORIGINS=["http://localhost:3000"]

# WebSocket Configuration
WEBSOCKET_TIMEOUT=30
ENABLE_REAL_TIME_LOGS=true
```

### Frontend Configuration

Update `frontend/src/App.tsx` if needed:

```typescript
const BACKEND_URL = 'http://localhost:8000';
const WEBSOCKET_URL = 'ws://localhost:8000/ws/logs';
```

## 📚 API Documentation

### Document Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/documents/` | List all documents with metadata |
| `GET` | `/api/documents/{id}/info` | Get document information |
| `GET` | `/api/documents/{id}/preview` | Preview document (PDF conversion) |
| `GET` | `/api/documents/{id}/original` | Get original file |
| `GET` | `/api/documents/{id}/download` | Download original file |
| `GET` | `/api/documents/{id}/download-pdf` | Download PDF version |
| `DELETE` | `/api/documents/{id}` | Delete document |

### File Upload & Progress

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload/documents` | Upload multiple files |
| `GET` | `/api/upload/status/{id}` | Get upload progress |
| `GET` | `/api/upload/history` | Upload history |
| `WebSocket` | `/ws/logs` | Real-time progress updates |

### RAG Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send message with RAG context |
| `POST` | `/api/chat/stream` | Streaming chat responses |
| `GET` | `/api/chat/history/{id}` | Conversation history |

### System Administration

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/admin/status` | System health check |
| `POST` | `/api/admin/reset` | Reset system data |
| `GET` | `/health` | Health check endpoint |

## 🛠️ Development

### Running Tests

```bash
# Backend tests
cd backend
python scripts/run_comprehensive_tests.py

# Frontend tests
cd frontend
npm test
```

### Building for Production

```bash
# Build frontend
cd frontend
npm run build

# Backend production setup
cd backend
pip install gunicorn
gunicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Adding New Document Types

1. Update `ALLOWED_EXTENSIONS` in `backend/app/core/config.py`
2. Add processing logic in `backend/app/services/document_service.py`
3. Update frontend validation in `frontend/src/components/FileUploadPanel.tsx`
4. Add file type icon in `getFileTypeIcon()` function

### System Cleanup

**Complete System Reset:**
```bash
# Comprehensive cleanup (recommended)
cd backend && ./scripts/cleanup_system.sh

# Options available:
./scripts/cleanup_system.sh --force        # No confirmation prompt
./scripts/cleanup_system.sh --quiet        # Minimal output
./scripts/cleanup_system.sh --skip-backup  # Skip file backups
```

**What Gets Cleaned:**
- ✅ All SQLite databases (including WAL files)
- ✅ ChromaDB vector storage (`./embeddings`)
- ✅ All uploaded and processed files
- ✅ All logs and cache files
- ✅ Parallel processing worker processes

**What Gets Preserved:**
- ✅ Virtual environment (`venv/`)
- ✅ Configuration files (`.env`)
- ✅ Test reports (`test_report_*.json`)
- ✅ Application code

## 🔍 Troubleshooting

### Common Issues

**Real-time progress not working:**
```bash
# Check WebSocket connection
curl -I -N -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Key: test" -H "Sec-WebSocket-Version: 13" http://localhost:8000/ws/logs

# Restart backend with WebSocket support
cd backend && source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Backend database errors:**
```bash
# Complete system reset (recommended)
cd backend && ./scripts/cleanup_system.sh --force

# Manual fix for "readonly database" errors
pkill -f "uvicorn app.main:app"  # Kill backend processes
pkill -f "loky-"  # Kill parallel workers
rm -f backend/data/*.db*  # Remove all database files
rm -f backend/data/*.wal backend/data/*.shm  # Remove WAL files
rm -rf backend/embeddings/*  # Clear vector database
mkdir -p backend/data backend/embeddings  # Recreate directories

# Check database permissions (if still having issues)
ls -la backend/data/
chmod 755 backend/data/
```

**Frontend build fails:**
```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

**Ollama connection issues:**
```bash
# Check Ollama status
ollama list

# Restart Ollama service
ollama serve

# Test model
ollama run llama3:8b-instruct-q8_0 "Hello"
```

**Document preview not working:**
```bash
# Check if documents API is accessible
curl http://localhost:8000/api/documents/

# Verify file permissions
ls -la backend/data/original_files/
ls -la backend/data/hashed_files/
```

### Performance Optimization

- **Processing Strategy Selection**: Enhanced algorithm for optimal performance
  - **Sequential**: Single files <10MB or batch total <5MB
  - **Parallel**: Multiple files >5MB total OR any file >10MB OR batch >50MB
- **Memory Usage**: Adjust `MAX_FILE_SIZE_MB` based on available RAM (default: 500MB)
- **CPU Usage**: Parallel processing uses `min(cpu_count(), 8)` cores
- **Database**: ChromaDB automatically optimizes vector storage
- **WebSocket**: Connection auto-reconnects if lost
- **Real-Time Progress**: All processing stages now provide live feedback

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- **Real-time Features**: Test WebSocket connections thoroughly
- **Progress Tracking**: Ensure all processing stages emit progress updates
- **Error Handling**: Provide clear error messages and recovery options
- **Mobile Support**: Test responsive design on various screen sizes
- **Performance**: Profile large file uploads and optimize as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for local AI model hosting
- **ChromaDB** for vector database capabilities
- **FastAPI** for the robust backend framework with WebSocket support
- **React** for the modern frontend experience
- **Tesseract OCR** for document text extraction
- **WebSocket API** for real-time communication

---

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

**Features requests and bug reports are welcome!**

**Happy chatting with your documents! 🚀📄** 