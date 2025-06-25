# 🤖 Personal Assistant AI Chatbot

A powerful desktop AI chatbot application with **local AI support using Ollama** that learns from your uploaded documents using advanced RAG (Retrieval Augmented Generation) architecture. **Complete privacy - everything runs locally on your machine!**

## ✨ Features

- 📄 **Advanced Document Processing**: Upload PDFs up to **500MB**, Word docs, text files, markdown, HTML
- 🖼️ **OCR for Scanned PDFs**: Automatic text extraction from image-based documents using Tesseract OCR
- ⚡ **True Multiprocessing**: Utilizes all CPU cores for parallel PDF processing (up to 8x speedup)
- 🧠 **RAG Architecture**: Retrieval Augmented Generation with vector embeddings
- 💬 **Context-Aware Chat**: AI remembers your documents and conversation history
- ✍️ **Content Generation**: Create emails, letters, blog posts, reports, summaries
- 🔍 **Semantic Search**: Find relevant information across all uploaded documents
- 🌍 **Multi-Language OCR**: Support for 100+ languages including English, Spanish, French, German, Chinese, Japanese, Arabic
- 🦙 **Local AI with Ollama**: Complete privacy, no data leaves your machine
- ⚛️ **Modern React Frontend**: Professional three-panel layout with drag & drop file upload
- 🎯 **Smart Processing**: Automatic detection of large files for parallel processing
- 🔒 **Privacy First**: All processing happens locally with Ollama - including OCR

## 🚀 Quick Start

### Option 1: React Frontend (Recommended) ⚛️
```bash
# 1. Clone and setup everything automatically
git clone <your-repo>
cd chatbot

# 2. Install Ollama and setup AI models
./setup_ollama.sh

# 3. Install OCR capabilities for scanned PDFs (optional)
./setup_ocr.sh

# 4. Start the React application
./start_react_app.sh
```

### Option 2: Basic Setup (No OCR) 🦙
```bash
# 1. Clone and setup everything automatically
git clone <your-repo>
cd chatbot

# 2. Install Ollama and setup AI models
./setup_ollama.sh

# 3. Start the React application
./start_react_app.sh
```

### Option 3: Manual Setup with OCR
```bash
# 1. Install Ollama and models
brew install ollama  # macOS
# or: curl -fsSL https://ollama.ai/install.sh | sh

ollama serve
ollama pull llama3:8b-instruct-q8_0

# 2. Install OCR system dependencies
brew install tesseract poppler tesseract-lang  # macOS
# or: sudo apt-get install tesseract-ocr poppler-utils  # Linux

# 3. Setup Python environment
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytesseract pdf2image PyMuPDF  # OCR packages

# 4. Configure environment
echo "LLM_PROVIDER=ollama" > .env
echo "OLLAMA_MODEL=llama3:8b-instruct-q8_0" >> .env
echo "MAX_FILE_SIZE_MB=500" >> .env
echo "ENABLE_OCR=true" >> .env

# 5. Start services
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
streamlit run ../streamlit_app.py --server.port 8501
```

### 🌐 Access Points
- **React Frontend**: http://localhost:3000 (Modern React UI)
- **Backend API**: http://localhost:8000 (FastAPI)
- **API Documentation**: http://localhost:8000/docs (Swagger UI)

## 🏗️ Architecture Overview

### Complete RAG System Architecture

```mermaid
graph TB
    subgraph "User Interface"
        UI["React Frontend<br/>localhost:3000"]
    end
    
    subgraph "API Layer"
        API["FastAPI Backend<br/>localhost:8000"]
        UPLOAD["Upload API<br/>/api/upload"]
        CHAT["Chat API<br/>/api/chat"]
        DOCS["Documents API<br/>/api/documents"]
    end
    
    subgraph "Document Processing Pipeline"
        DOC_SERVICE["Document Service<br/>PDF/DOCX/TXT/HTML Parser"]
        TEXT_SPLIT["Text Splitter<br/>Chunking Strategy"]
        EMBED_GEN["Embedding Generator<br/>SentenceTransformers"]
    end
    
    subgraph "Storage Layer"
        VECTOR_DB[("ChromaDB<br/>Vector Embeddings")]
        SQL_DB[("SQLite<br/>Metadata & History")]
        FILE_STORE["File Storage<br/>./data/"]
    end
    
    subgraph "AI Processing"
        OLLAMA["Ollama Service<br/>localhost:11434"]
        LLM["Llama3:8b-instruct-q8_0<br/>Local AI Model"]
        RAG_ENGINE["RAG Engine<br/>Context Retrieval"]
    end
    
    subgraph "Services"
        VECTOR_SVC["Vector Service<br/>Similarity Search"]
        CHAT_SVC["Chat Service<br/>Conversation Management"]
        DB_SVC["Database Service<br/>Metadata Storage"]
    end

    %% User Flow
    UI --> API
    
    %% Document Upload Flow
    API --> UPLOAD
    UPLOAD --> DOC_SERVICE
    DOC_SERVICE --> TEXT_SPLIT
    TEXT_SPLIT --> EMBED_GEN
    EMBED_GEN --> VECTOR_DB
    DOC_SERVICE --> SQL_DB
    DOC_SERVICE --> FILE_STORE
    
    %% Chat Flow
    API --> CHAT
    CHAT --> CHAT_SVC
    CHAT_SVC --> RAG_ENGINE
    RAG_ENGINE --> VECTOR_SVC
    VECTOR_SVC --> VECTOR_DB
    RAG_ENGINE --> OLLAMA
    OLLAMA --> LLM
    CHAT_SVC --> SQL_DB
    
    %% Document Management
    API --> DOCS
    DOCS --> DB_SVC
    DB_SVC --> SQL_DB
    DB_SVC --> VECTOR_SVC

    style UI fill:#1e3a8a,color:#ffffff
    style OLLAMA fill:#dc2626,color:#ffffff
    style LLM fill:#dc2626,color:#ffffff
    style VECTOR_DB fill:#b45309,color:#ffffff
    style SQL_DB fill:#b45309,color:#ffffff
    style RAG_ENGINE fill:#166534,color:#ffffff
    style DOC_SERVICE fill:#7c2d12,color:#ffffff
```

## ⚡ Multiprocessing Architecture

### True Parallel Processing with CPU Cores

The system uses **multiprocessing** (not just async) to achieve true parallelism across CPU cores:

#### **🖥️ CPU Core Utilization**
- **Automatic Detection**: Uses `min(cpu_count(), 8)` cores for optimal performance
- **Process Pool**: Dedicated `ProcessPoolExecutor` for CPU-intensive tasks
- **Load Balancing**: Automatic distribution of PDF chunks across available cores
- **Memory Isolation**: Each process has isolated memory space for stability

#### **📈 Performance Benefits**
- **PDF Processing**: Up to **8x speedup** for large documents
- **OCR Processing**: Parallel text recognition across multiple pages
- **Text Chunking**: Concurrent processing of extracted text
- **Embedding Generation**: Batch processing with multicore support

#### **🎯 Smart Processing Logic**
```python
# Automatic parallel processing triggers:
if file_size > 10MB or estimated_time > 30s:
    # Split PDF into chunks
    # Process chunks across CPU cores
    # Combine results efficiently
```

#### **🔥 Real-Time Progress Tracking**
- **Per-Core Status**: Individual progress for each CPU core
- **Neon-Themed UI**: Beautiful visual indicators for parallel processing
- **Time Estimates**: Accurate ETA based on multicore performance
- **Chunk Status**: Real-time updates for each parallel task

## 📊 Data Processing Pipeline

### Multi-Level Parallel Processing Architecture

The system performs **true parallelism at 4 distinct levels** using multiprocessing:

```mermaid
graph TD
    subgraph "Multi-Level Parallel Processing"
        A["📄 Large PDF Upload<br/>200MB, 160 pages"]
        
        subgraph "Level 1: PDF Chunking"
            B["🔪 Adaptive Splitting<br/>4 chunks × 40 pages"]
            C1["📋 Chunk 1<br/>Pages 1-40"]
            C2["📋 Chunk 2<br/>Pages 41-80"] 
            C3["📋 Chunk 3<br/>Pages 81-120"]
            C4["📋 Chunk 4<br/>Pages 121-160"]
        end
        
        subgraph "Level 2: Text Extraction (4 CPU Cores)"
            D["⚡ ProcessPoolExecutor"]
            E1["🔧 Core 1<br/>OCR + Extract"]
            E2["🔧 Core 2<br/>OCR + Extract"]
            E3["🔧 Core 3<br/>OCR + Extract"] 
            E4["🔧 Core 4<br/>OCR + Extract"]
        end
        
        subgraph "Level 3: Text Chunking (4 CPU Cores)"
            F["⚡ Parallel Chunking"]
            G1["🧩 Core 1<br/>18 chunks"]
            G2["🧩 Core 2<br/>15 chunks"]
            G3["🧩 Core 3<br/>22 chunks"]
            G4["🧩 Core 4<br/>16 chunks"]
        end
        
        subgraph "Level 4: Embedding Generation (4 CPU Cores)"
            H["⚡ Parallel Embeddings"]
            I1["🧠 Core 1<br/>Batch 1-18"]
            I2["🧠 Core 2<br/>Batch 19-33"]
            I3["🧠 Core 3<br/>Batch 34-55"]
            I4["🧠 Core 4<br/>Batch 56-71"]
        end
        
        J["💾 Parallel Storage<br/>ChromaDB + SQLite"]
        K["✅ Result: 71 chunks<br/>⚡ 6.4x speedup"]
    end
    
    A --> B
    B --> C1 & C2 & C3 & C4
    C1 --> D
    C2 --> D
    C3 --> D 
    C4 --> D
    D --> E1 & E2 & E3 & E4
    E1 --> F
    E2 --> F
    E3 --> F
    E4 --> F
    F --> G1 & G2 & G3 & G4
    G1 --> H
    G2 --> H
    G3 --> H
    G4 --> H
    H --> I1 & I2 & I3 & I4
    I1 --> J
    I2 --> J
    I3 --> J
    I4 --> J
    J --> K
    
    style A fill:#dc2626,color:#fff
    style D fill:#059669,color:#fff
    style F fill:#7c3aed,color:#fff
    style H fill:#db2777,color:#fff
    style J fill:#0891b2,color:#fff
    style K fill:#16a34a,color:#fff
```

#### **🚀 4-Level Parallel Processing Benefits**

1. **📄 Level 1 - Adaptive PDF Splitting**: 
   - Smart chunking based on document characteristics
   - Memory and time-optimized chunk sizes (5-50 pages)
   - Load balancing across available CPU cores

2. **🔧 Level 2 - Parallel Text Extraction**: 
   - True multiprocessing with ProcessPoolExecutor
   - Each CPU core processes different PDF chunks simultaneously
   - OCR processing distributed across cores for scanned documents

3. **🧩 Level 3 - Parallel Text Chunking**: 
   - Concurrent creation of embedding-ready text chunks
   - LangChain RecursiveCharacterTextSplitter in parallel
   - Memory-efficient processing of extracted text

4. **🧠 Level 4 - Parallel Embedding Generation**: 
   - Batch embedding creation across multiple cores
   - SentenceTransformers distributed processing
   - Simultaneous vector generation and ChromaDB storage

### Enhanced Document Processing with OCR Support

The system now features **two processing pipelines** depending on file size and type:

#### **🔍 OCR-Enhanced Pipeline with Parallel Processing (Large/Scanned Files)**

```mermaid
graph TD
    subgraph "File Analysis & Routing"
        UPLOAD["📤 File Upload<br/>Up to 500MB"]
        ANALYZE["🔍 Smart Analysis<br/>Size, OCR needs, CPU cores"]
        DECISION{"⚡ Processing Strategy<br/>Parallel vs Standard"}
    end
    
    subgraph "Parallel OCR Processing Engine"
        SPLIT_PDF["🔪 PDF Splitting<br/>20 pages per chunk"]
        PROCESS_POOL["⚙️ Process Pool<br/>8 CPU Cores Active"]
        
        subgraph "Parallel Workers"
            WORKER1["🔧 Core 1<br/>Pages 1-20<br/>OCR + Extract"]
            WORKER2["🔧 Core 2<br/>Pages 21-40<br/>OCR + Extract"]
            WORKER3["🔧 Core 3<br/>Pages 41-60<br/>OCR + Extract"]
            WORKER4["🔧 Core 4<br/>Pages 61-80<br/>OCR + Extract"]
            WORKER5["🔧 Core 5<br/>Pages 81-100<br/>OCR + Extract"]
            WORKER6["🔧 Core 6<br/>Pages 101-120<br/>OCR + Extract"]
            WORKER7["🔧 Core 7<br/>Pages 121-140<br/>OCR + Extract"]
            WORKER8["🔧 Core 8<br/>Pages 141-160<br/>OCR + Extract"]
        end
        
        MERGE_RESULTS["🔗 Intelligent Merge<br/>Page order preservation"]
    end
    
    subgraph "Standard OCR Processing"
        EXTRACT_TRY["🔍 Text Extraction<br/>PyMuPDF → PyPDF2"]
        OCR_DETECT{"🖼️ OCR Detection<br/>Text < 100 chars?"}
        PDF2IMG["📸 PDF → Images<br/>pdf2image (300 DPI)"]
        TESSERACT["🔤 Tesseract OCR<br/>Multi-language"]
    end
    
    subgraph "Memory-Efficient Post-Processing"
        CHUNK_BATCH["✂️ Parallel Chunking<br/>1000 chars + 200 overlap"]
        EMBED_PARALLEL["🧮 Parallel Embeddings<br/>Multi-core processing"]
        REAL_TIME["📊 Real-time Progress<br/>Per-core status + ETA"]
    end
    
    subgraph "High-Performance Storage"
        CHROMADB[("🗄️ ChromaDB<br/>Vector Embeddings")]
        SQLITE[("📋 SQLite<br/>Processing Metadata")]
        FILES[("💾 Distributed Storage<br/>./data/ + chunks")]
    end

    UPLOAD --> ANALYZE
    ANALYZE --> DECISION
    
    %% Parallel Path
    DECISION -->|"Large files > 10MB<br/>OR > 30s estimate"| SPLIT_PDF
    SPLIT_PDF --> PROCESS_POOL
    PROCESS_POOL --> WORKER1
    PROCESS_POOL --> WORKER2
    PROCESS_POOL --> WORKER3
    PROCESS_POOL --> WORKER4
    PROCESS_POOL --> WORKER5
    PROCESS_POOL --> WORKER6
    PROCESS_POOL --> WORKER7
    PROCESS_POOL --> WORKER8
    
    WORKER1 --> MERGE_RESULTS
    WORKER2 --> MERGE_RESULTS
    WORKER3 --> MERGE_RESULTS
    WORKER4 --> MERGE_RESULTS
    WORKER5 --> MERGE_RESULTS
    WORKER6 --> MERGE_RESULTS
    WORKER7 --> MERGE_RESULTS
    WORKER8 --> MERGE_RESULTS
    
    %% Standard Path
    DECISION -->|"Small files < 10MB"| EXTRACT_TRY
    EXTRACT_TRY --> OCR_DETECT
    OCR_DETECT -->|"Sufficient text"| CHUNK_BATCH
    OCR_DETECT -->|"OCR needed"| PDF2IMG
    PDF2IMG --> TESSERACT
    TESSERACT --> CHUNK_BATCH
    
    %% Convergence
    MERGE_RESULTS --> CHUNK_BATCH
    CHUNK_BATCH --> EMBED_PARALLEL
    EMBED_PARALLEL --> REAL_TIME
    REAL_TIME --> CHROMADB
    REAL_TIME --> SQLITE
    REAL_TIME --> FILES

    %% Styling
    style UPLOAD fill:#dc2626,color:#ffffff
    style PROCESS_POOL fill:#059669,color:#ffffff
    style WORKER1 fill:#7c3aed,color:#ffffff
    style WORKER2 fill:#7c3aed,color:#ffffff
    style WORKER3 fill:#7c3aed,color:#ffffff
    style WORKER4 fill:#7c3aed,color:#ffffff
    style WORKER5 fill:#7c3aed,color:#ffffff
    style WORKER6 fill:#7c3aed,color:#ffffff
    style WORKER7 fill:#7c3aed,color:#ffffff
    style WORKER8 fill:#7c3aed,color:#ffffff
    style REAL_TIME fill:#f59e0b,color:#ffffff
    style CHROMADB fill:#1e40af,color:#ffffff
    style SQLITE fill:#1e40af,color:#ffffff
    style MERGE_RESULTS fill:#059669,color:#ffffff
```

### **🚀 Parallel Processing Capabilities**

#### **Intelligent Processing Decision**
```python
# Automatic parallel processing triggers:
if file_size > 10MB or estimated_processing_time > 30s:
    # Route to parallel processing engine
    # Split PDF into 20-page chunks
    # Distribute across available CPU cores
    # Process with true multiprocessing
```

#### **Per-Core OCR Processing**
- **Core 1-8**: Each processes 20-page PDF chunks simultaneously
- **OCR Pipeline per Core**: 
  - PDF chunk → Images (pdf2image)
  - Images → Text (Tesseract OCR)
  - Text → Cleaned output
  - Progress → Real-time status updates

#### **Performance Optimization**
- **8x Speedup**: Measured performance improvement on large files
- **Memory Isolation**: Each process has dedicated memory space
- **Load Balancing**: Automatic distribution of workload
- **Fault Tolerance**: Failed chunks don't affect other cores

#### **Real-Time Monitoring**
- **Neon Progress Bars**: Beautiful visual feedback per core
- **ETA Calculations**: Accurate time estimates based on multicore performance
- **CPU Utilization**: Live display of core usage
- **Chunk Status**: Individual progress for each PDF section



### How Documents Are Processed by Llama3

When you upload a document, here's exactly what happens:

#### **📋 Processing Stages (7-Stage Pipeline)**

1. **📤 File Upload & Validation**: Document uploaded via Streamlit UI
   - File size check (up to 500MB)
   - Format validation (PDF, DOCX, TXT, HTML, MD)
   - Content type detection

2. **🔍 Smart Format Detection**: System detects processing route
   - **Large files (>50MB)** → Enhanced OCR pipeline
   - **PDF files** → OCR-capable pipeline (auto-detection)
   - **Regular files** → Standard pipeline

3. **📝 Intelligent Text Extraction**: 
   - **PDF**: PyMuPDF (efficient) → PyPDF2 (fallback)
   - **DOCX**: python-docx reads paragraphs
   - **TXT/MD**: Direct text with encoding detection
   - **HTML**: BeautifulSoup strips tags and extracts content

4. **🖼️ OCR Detection & Processing**: 
   - If text extraction yields < 100 characters → **OCR needed**
   - **pdf2image**: Convert PDF pages to high-resolution images (300 DPI)
   - **Tesseract OCR**: Extract text from images (100+ languages supported)
   - **Text merging**: Combine OCR results from all pages

5. **✂️ Memory-Efficient Text Chunking**: 
   - **LangChain RecursiveCharacterTextSplitter** creates overlapping chunks
   - Default: 1000 characters per chunk with 200 character overlap
   - **Batch processing** for large documents (>1MB text)
   - Preserves context between chunks

6. **🧮 Batch Vector Embeddings**: 
   - **SentenceTransformers** creates embeddings for each chunk
   - **Batch processing**: 100 chunks at a time for memory efficiency
   - Progress tracking for large documents

7. **💾 Multi-Tier Storage**: 
   - **Vector embeddings** → ChromaDB for similarity search
   - **Document metadata** → SQLite for management (including OCR info)
   - **Original files** → Local file system (./data/)

### RAG (Retrieval Augmented Generation) Process

When you ask a question:

1. **❓ Query Processing**: Your question is converted to embeddings
2. **🔍 Similarity Search**: ChromaDB finds most relevant document chunks
3. **📋 Context Assembly**: Top 5 relevant chunks are combined as context
4. **🎯 Prompt Building**: Context + conversation history + question → structured prompt
5. **🤖 AI Generation**: Llama3 model generates response using retrieved context
6. **💬 Response**: Contextual answer referencing your documents

## 📁 Project Structure

```
chatbot/
├── streamlit_app.py              # Main Streamlit desktop application
├── setup_ollama.sh               # Automated Ollama setup script
├── start_desktop_app.sh          # One-click startup script
├── backend/                      # FastAPI backend service
│   ├── app/
│   │   ├── api/routes/           # API endpoint definitions
│   │   │   ├── chat.py           # Chat conversations with RAG
│   │   │   ├── upload.py         # Document upload and processing
│   │   │   ├── documents.py      # Document management
│   │   │   └── generate.py       # Content generation
│   │   ├── core/                 # Core configuration and prompts
│   │   │   ├── config.py         # Application settings
│   │   │   └── prompts.py        # AI prompt templates
│   │   ├── models/               # Data models and schemas
│   │   │   └── schemas.py        # Pydantic data models
│   │   ├── services/             # Business logic services
│   │   │   ├── document_service.py    # Document processing pipeline
│   │   │   ├── chat_service.py         # RAG chat logic
│   │   │   ├── vector_service.py       # ChromaDB operations
│   │   │   ├── database_service.py     # SQLite operations
│   │   │   └── web_scraper.py          # URL content extraction
│   │   └── main.py               # FastAPI application entry
│   ├── data/                     # Uploaded files storage
│   ├── embeddings/               # ChromaDB vector database
│   ├── requirements.txt          # Complete Python dependencies (includes OCR)
│   └── .env                      # Configuration file
└── README.md                     # This comprehensive guide
```

## 🔧 Technology Stack

### **Backend Services**
- **FastAPI**: High-performance async Python web framework
- **Ollama**: Local AI model server (llama3:8b-instruct-q8_0)
- **ChromaDB**: Vector database for semantic search
- **SQLite**: Document metadata and conversation history
- **LangChain**: Document processing and LLM orchestration

### **Enhanced Document Processing**
- **PyMuPDF (fitz)**: High-performance PDF processing for large files
- **PyPDF2**: PDF text extraction (fallback)
- **python-docx**: Word document processing  
- **BeautifulSoup**: HTML parsing and cleaning
- **SentenceTransformers**: Text embedding generation
- **RecursiveCharacterTextSplitter**: Intelligent text chunking

### **OCR & Image Processing**
- **Tesseract OCR**: Multi-language optical character recognition
- **pdf2image**: PDF to image conversion for OCR processing
- **Pillow (PIL)**: Image processing and manipulation
- **Poppler**: PDF rendering utilities for image conversion

### **Memory & Performance Optimization**
- **aiofiles**: Async file I/O for large file handling
- **asyncio**: Asynchronous processing for concurrent operations
- **psutil**: System resource monitoring
- **batch processing**: Memory-efficient chunk processing

### **Frontend**
- **Streamlit**: Modern Python-based desktop UI with progress tracking
- **Streamlit-Chat**: Enhanced chat components
- **Real-time progress**: 7-stage upload progress visualization

### **AI Models**
- **Llama3:8b-instruct-q8_0**: 8B parameter instruction-tuned model
- **all-MiniLM-L6-v2**: Sentence embedding model for vector search
- **Multi-language support**: 100+ languages via Tesseract

## ⚙️ Configuration

### Environment Variables (.env file)

```bash
# AI Provider Configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b-instruct-q8_0

# Enhanced File Processing Configuration
MAX_FILE_SIZE_MB=500                    # Maximum file size (500MB)
ENABLE_OCR=true                         # Enable OCR for scanned PDFs
OCR_LANGUAGE=eng                        # Primary OCR language
OCR_LANGUAGES_MULTI=eng+spa+fra         # Multi-language OCR support
OCR_DPI=300                            # Image resolution for OCR
PROCESS_LARGE_FILES_ASYNC=true         # Async processing for large files

# Database Configuration
CHROMA_DB_PATH=./embeddings
CHROMA_COLLECTION_NAME=personal_assistant_docs
DATA_STORAGE_PATH=./data

# RAG Configuration
CHUNK_SIZE=1000                        # Text chunk size
CHUNK_OVERLAP=200                      # Overlap between chunks
MAX_SEARCH_RESULTS=5                   # Top-K retrieval
EMBEDDING_MODEL=all-MiniLM-L6-v2      # Embedding model
BATCH_SIZE_EMBEDDINGS=100              # Batch size for large documents

# Memory Management
ENABLE_BATCH_PROCESSING=true           # Batch processing for large files
MAX_MEMORY_USAGE_MB=4096              # Maximum memory usage
CLEANUP_TEMP_FILES=true               # Auto-cleanup temporary files

# Chat Configuration
MAX_CHAT_HISTORY=10
DEFAULT_TEMPERATURE=0.7

# Progress Tracking
ENABLE_PROGRESS_TRACKING=true         # Real-time upload progress
PROGRESS_UPDATE_INTERVAL=5            # Progress update frequency (seconds)

# Optional: Enable debug mode
DEBUG=true
LOG_LEVEL=INFO                        # INFO, DEBUG, WARNING, ERROR
```

### Supported File Formats & Processing Methods

| Format | Extensions | Max Size | Processing Method | OCR Support |
|--------|------------|----------|-------------------|-------------|
| **PDF (Text)** | `.pdf` | 500MB | PyMuPDF → PyPDF2 fallback | ⚡ Auto-detection |
| **PDF (Scanned)** | `.pdf` | 500MB | pdf2image → Tesseract OCR | ✅ Full support |
| **Word Documents** | `.docx`, `.doc` | 500MB | python-docx paragraph extraction | ❌ Text-based |
| **Text Files** | `.txt`, `.md` | 500MB | Direct text with encoding detection | ❌ Text-based |
| **HTML Pages** | `.html`, `.htm` | 500MB | BeautifulSoup tag stripping | ❌ Text-based |
| **Web Content** | URLs | N/A | Web scraping via aiohttp | ❌ Text-based |

#### **🔍 OCR Language Support**

**Automatically supported languages** (100+ total):
- 🇺🇸 **English** (eng) - Primary
- 🇪🇸 **Spanish** (spa) 
- 🇫🇷 **French** (fra)
- 🇩🇪 **German** (deu)
- 🇨🇳 **Chinese Simplified** (chi_sim)
- 🇯🇵 **Japanese** (jpn)
- 🇰🇷 **Korean** (kor)
- 🇦🇪 **Arabic** (ara)
- 🇷🇺 **Russian** (rus)
- 🇮🇳 **Hindi** (hin)

*Configure OCR language in `.env`: `OCR_LANGUAGE=eng+spa+fra` for multi-language documents*

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.9+** (Python 3.11+ recommended)
- **8GB RAM minimum** (for Llama3:8b model)
- **10GB free disk space** (for models and data)

### 🦙 Ollama Setup (Complete Guide)

#### Quick Installation
```bash
# macOS
brew install ollama

# Linux/WSL
curl -fsSL https://ollama.ai/install.sh | sh

# Start service
ollama serve

# Install the specific model used by the chatbot
ollama pull llama3:8b-instruct-q8_0
```

#### Model Options
```bash
# Recommended (8GB RAM)
ollama pull llama3:8b-instruct-q8_0

# Alternatives
ollama pull llama3:latest          # Latest Llama3 (larger)
ollama pull mistral               # Alternative model
ollama pull codellama             # Code-specialized
ollama pull gemma:7b              # Google's Gemma
```

#### Verify Setup
```bash
# Check service
curl http://localhost:11434/api/tags

# Test model
ollama run llama3:8b-instruct-q8_0 "Hello, how are you?"
```

### Python Environment Setup

#### Standard Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### All Platforms (including macOS)
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # Handles macOS compatibility automatically
```

### Manual Configuration
```bash
# Create environment file
cat > backend/.env << EOF
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b-instruct-q8_0
CHROMA_DB_PATH=./embeddings
DATA_STORAGE_PATH=./data
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_SEARCH_RESULTS=5
EOF
```

## 🚀 Usage Guide

### Starting the Application

#### Automated Startup
```bash
./start_desktop_app.sh
```

#### Manual Startup
```bash
# Terminal 1: Backend API
cd backend && source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit UI
cd backend && source venv/bin/activate
streamlit run ../streamlit_app.py --server.port 8501
```

### Using the Chatbot

1. **📤 Upload Documents**:
   - Click "Upload Documents" in sidebar
   - Select PDFs, Word docs, text files
   - Wait for processing completion

2. **💬 Start Chatting**:
   - Ask questions about your documents
   - AI will reference uploaded content
   - Conversations are saved locally

3. **✍️ Generate Content**:
   - Use "Content Generation" tab
   - Create emails, reports, summaries
   - Based on your uploaded documents

4. **🔍 Search Documents**:
   - Use semantic search to find information
   - Results show relevant document chunks
   - Click to see full context

### API Usage

#### Upload Document
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "files=@document.pdf"
```

#### Chat with Context
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What does the document say about AI?", "use_context": true}'
```

#### List Documents
```bash
curl "http://localhost:8000/api/documents"
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. **Ollama Connection Failed** 🦙
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve

# Verify model exists
ollama list | grep llama3
```

#### 2. **Model Not Found**
```bash
# Pull the required model
ollama pull llama3:8b-instruct-q8_0

# Update config if using different model
echo "OLLAMA_MODEL=your_model_name" >> backend/.env
```

#### 3. **Python Dependencies Error**
```bash
# macOS lxml compilation issue
pip install -r requirements-macos.txt

# Or install system dependencies
brew install libxml2 libxslt

# Permission issues
pip install --user -r requirements.txt
```

#### 4. **Memory Issues**
```bash
# Use smaller model
ollama pull phi3  # Only 2.3GB
echo "OLLAMA_MODEL=phi3" >> backend/.env

# Check system memory
free -h  # Linux
vm_stat  # macOS
```

#### 5. **Document Processing Failed**
- Check file format is supported
- Ensure file isn't corrupted
- Try smaller files first
- Check logs: `tail -f backend/app.log`

#### 6. **Vector Database Issues**
```bash
# Reset ChromaDB if needed
rm -rf backend/embeddings
# Restart application to recreate
```

#### 7. **OCR Processing Issues** 🔍
```bash
# Verify Tesseract installation
tesseract --version
tesseract --list-langs

# Test OCR manually
tesseract input.png output.txt

# Check Poppler installation
pdftoppm -h

# Verify Python OCR packages
python -c "import pytesseract, pdf2image, fitz; print('OCR packages OK')"
```

#### 8. **Large File Processing Failures**
```bash
# Check available memory
free -h  # Linux
vm_stat  # macOS

# Monitor processing in real-time
tail -f backend/app.log

# Reduce file size or increase RAM
# Split large PDFs into smaller chunks if needed
```

#### 9. **OCR Language Issues**
```bash
# Install additional language packs
brew install tesseract-lang  # macOS (all languages)
sudo apt-get install tesseract-ocr-spa  # Spanish
sudo apt-get install tesseract-ocr-fra  # French

# Configure multi-language OCR
echo "OCR_LANGUAGE=eng+spa+fra" >> backend/.env
```

### Performance Optimization

#### For Large Documents
- Documents automatically chunked (1000 chars with 200 overlap)
- Async processing for multiple uploads
- Vector embeddings cached for fast retrieval

#### Memory Management
- ChromaDB handles vector storage efficiently
- SQLite for lightweight metadata
- Conversation history limited to prevent memory issues

#### Performance Expectations for Large Files

| File Size | Type | Pages | OCR Time | Total Time | Memory Usage | Chunks Generated |
|-----------|------|-------|----------|------------|--------------|------------------|
| **50MB** | Text PDF | 25 | N/A | 15s | 1.5GB | ~150 |
| **50MB** | Scanned PDF | 25 | 30s | 45s | 2GB | ~120 |
| **100MB** | Text PDF | 50 | N/A | 25s | 2.5GB | ~300 |
| **100MB** | Scanned PDF | 50 | 60s | 90s | 3GB | ~240 |
| **200MB** | Text PDF | 100 | N/A | 45s | 3.5GB | ~600 |
| **200MB** | Scanned PDF | 100 | 120s | 180s | 4GB | ~480 |
| **500MB** | Text PDF | 250 | N/A | 120s | 6GB | ~1500 |
| **500MB** | Scanned PDF | 250 | 300s | 450s | 8GB | ~1200 |

*Performance varies based on document complexity, image quality, and hardware specifications*

#### Hardware Recommendations

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **RAM** | 8GB | 16GB | 32GB+ |
| **Storage** | 10GB | 50GB | 100GB+ |
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **GPU** | None | Optional | RTX 3080+ |

#### **🚀 Performance Optimization Tips**

- **For 200MB+ files**: Ensure 16GB+ RAM for optimal performance
- **OCR processing**: Multi-core CPUs significantly reduce processing time
- **SSD storage**: Faster I/O improves large file handling
- **Close other applications**: Free up memory for large document processing
- **Batch uploads**: Process multiple smaller files instead of one massive file when possible

## 🔒 Privacy & Security

### Data Privacy Features
- **100% Local Processing**: All AI inference happens on your machine
- **No External API Calls**: When using Ollama (vs OpenAI)
- **Local Storage**: Documents stored in `./data/` directory
- **Conversation History**: Saved locally in SQLite database
- **Vector Embeddings**: Stored locally in ChromaDB

### Security Measures
- Input validation and sanitization
- File type restrictions and size limits
- No network data transmission (Ollama mode)
- Local authentication (single-user desktop app)

## 🗺️ Development Roadmap

### **Phase 1: Core Features** ✅
- ✅ Document upload and processing (PDF, DOCX, TXT, HTML)
- ✅ RAG architecture with ChromaDB
- ✅ Local AI support with Ollama
- ✅ Streamlit desktop UI
- ✅ Conversation management
- ✅ Content generation

### **Phase 2: Enhanced Features** 🚧
- 🔄 Advanced search with filters
- 🔄 Document summarization
- 🔄 Multi-language support
- 🔄 Voice interface
- 🔄 Export conversations

### **Phase 3: Advanced Features** 📋
- 📋 Multi-modal support (images, audio)
- 📋 Collaborative features
- 📋 API authentication
- 📋 Plugin system
- 📋 Mobile app

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Add type hints for new functions
- Include docstrings for public methods
- Write tests for new features
- Update documentation for changes

## 📝 License

MIT License - see LICENSE file for details

## 🙋‍♂️ Support & FAQ

### Frequently Asked Questions

**Q: Do I need an internet connection?**
A: Only for initial setup. Once Ollama and models are installed, everything runs offline.

**Q: Can I use different AI models?**
A: Yes! Install any Ollama model and update `OLLAMA_MODEL` in `.env`

**Q: What file sizes are supported?**
A: No hard limit, but large files (>100MB) may take longer to process.

**Q: Is my data secure?**
A: Yes! With Ollama, everything processes locally. No data leaves your machine.

**Q: Can I run this on a low-end machine?**
A: Use `phi3` model (2.3GB) for machines with 4-8GB RAM.

**Q: How do I backup my data?**
A: Copy `backend/data/`, `backend/embeddings/`, and conversation database.

**Q: Can I process multiple large files simultaneously?**
A: Yes, but ensure sufficient RAM (16GB+ recommended for multiple 200MB+ files).

**Q: What happens if OCR fails?**
A: The system gracefully falls back to standard text extraction and provides detailed error messages.

**Q: Can I disable OCR for faster processing?**
A: Yes, set `ENABLE_OCR=false` in `.env` to skip OCR processing entirely.

## 🔍 Database Inspection & Management

### **Inspect Your Data** 📊

The chatbot stores data in two main databases:
- **SQLite** (`backend/data/chatbot.db`) - Document metadata, chat history, sessions
- **ChromaDB** (`backend/embeddings/chroma.sqlite3`) - Vector embeddings and full-text search

#### **Quick Database Overview**
```bash
# Run comprehensive database inspection
python inspect_databases.py

# Quick SQL queries on main database
python quick_sql.py --query "SELECT filename, doc_type, upload_date FROM documents ORDER BY upload_date DESC LIMIT 5;"

# Query ChromaDB internal storage
python quick_sql.py --chroma --query "SELECT COUNT(*) FROM embeddings;"
```

#### **VS Code Database Integration** 🎯

1. **Install SQLite Extension**:
   - Open VS Code
   - Install "SQLite Viewer" by qwtel
   - Or install "SQLite" by alexcvzz

2. **Connect to Databases**:
   ```bash
   # In VS Code Command Palette (Cmd+Shift+P):
   # > SQLite: Open Database
   
   # Select database files:
   backend/data/chatbot.db          # Main application data
   backend/embeddings/chroma.sqlite3 # Vector embeddings data
   ```

3. **Useful SQL Queries**:
   ```sql
   -- List all uploaded documents
   SELECT id, filename, upload_date, file_size FROM documents ORDER BY upload_date DESC;
   
   -- Recent chat conversations
   SELECT id, user_message, ai_response, timestamp FROM conversations ORDER BY timestamp DESC LIMIT 10;
   
   -- Document statistics
   SELECT doc_type, COUNT(*) as count, SUM(file_size) as total_size FROM documents GROUP BY doc_type;
   
   -- Large documents (>1MB)
   SELECT filename, file_size, upload_date FROM documents WHERE file_size > 1000000 ORDER BY file_size DESC;
   
   -- ChromaDB collection info
   SELECT name, dimension FROM collections;
   
   -- Vector embedding count
   SELECT COUNT(*) as total_embeddings FROM embeddings;
   
   -- Full-text search data
   SELECT COUNT(*) as searchable_chunks FROM embedding_fulltext_search;
   ```

#### **Interactive SQL Tool** 💻
```bash
# Start interactive SQL session
python quick_sql.py

# Example session:
SQL> SELECT COUNT(*) FROM documents;
SQL> SELECT filename FROM documents WHERE doc_type = 'pdf';
SQL> SELECT * FROM sessions ORDER BY created_at DESC LIMIT 3;
SQL> exit
```

#### **Export Data to CSV** 📤
```bash
# Export for analysis in Excel/Google Sheets
python inspect_databases.py  # Creates CSV files automatically

# Files created:
# - documents_export.csv (all document metadata)
# - chat_history_export.csv (conversation history)
```

#### **Database Schema Overview**

**Main Database (`chatbot.db`)**:
```sql
-- Documents table
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    content TEXT,
    doc_type TEXT,
    upload_date DATETIME,
    metadata TEXT
);

-- Conversations table
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    user_message TEXT,
    ai_response TEXT,
    sources TEXT,
    timestamp DATETIME,
    metadata TEXT
);

-- Sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    created_at DATETIME,
    last_activity DATETIME,
    metadata TEXT
);
```

**ChromaDB Schema** (Internal):
- `collections` - Vector collections metadata
- `embeddings` - Vector embeddings storage
- `embedding_metadata` - Chunk metadata (filename, doc_type, etc.)
- `embedding_fulltext_search` - Full-text search index

#### **Data Analysis Examples** 📈

```bash
# Find documents with most chunks
python quick_sql.py --chroma --query "
SELECT 
    string_value as filename,
    COUNT(*) as chunk_count 
FROM embedding_metadata 
WHERE key = 'filename' 
GROUP BY string_value 
ORDER BY chunk_count DESC 
LIMIT 10;"

# Check processing performance
python quick_sql.py --query "
SELECT 
    filename,
    file_size,
    JSON_EXTRACT(metadata, '$.chunk_count') as chunks,
    upload_date
FROM documents 
ORDER BY file_size DESC;"

# Find most recent conversations
python quick_sql.py --query "
SELECT 
    user_message,
    LENGTH(ai_response) as response_length,
    timestamp
FROM conversations 
ORDER BY timestamp DESC 
LIMIT 5;"
```

#### **Database Maintenance** 🔧

```bash
# Check database integrity
sqlite3 backend/data/chatbot.db "PRAGMA integrity_check;"

# Vacuum databases (reclaim space)
sqlite3 backend/data/chatbot.db "VACUUM;"
sqlite3 backend/embeddings/chroma.sqlite3 "VACUUM;"

# Backup databases
cp backend/data/chatbot.db backup_$(date +%Y%m%d).db
cp -r backend/embeddings/ embeddings_backup_$(date +%Y%m%d)/
```

#### **Troubleshooting Database Issues** 🚨

```bash
# Check if databases exist
ls -la backend/data/chatbot.db
ls -la backend/embeddings/chroma.sqlite3

# Check database permissions
chmod 664 backend/data/chatbot.db

# Reset databases (CAUTION: Deletes all data)
rm backend/data/chatbot.db
rm -rf backend/embeddings/
# Restart application to recreate
```

#### **Database Performance Monitoring** ⚡

```bash
# Monitor database size
du -h backend/data/chatbot.db
du -h backend/embeddings/

# Check ChromaDB collection statistics
python -c "
import chromadb
client = chromadb.PersistentClient(path='backend/embeddings')
collection = client.get_collection('documents')
print(f'Total documents: {collection.count()}')
print(f'Collection metadata: {collection.metadata}')
"
```

## 🧪 Testing Your OCR Setup

### **Quick OCR Test**
```bash
# Test OCR system capabilities
cd backend && python -c "
import pytesseract
import pdf2image
print('✅ OCR System Status:')
print(f'Tesseract version: {pytesseract.get_tesseract_version()}')
print(f'Available languages: {pytesseract.get_languages()[:10]}')
print('🎯 Ready for 500MB scanned PDFs!')
"
```

### **End-to-End Test**
1. **Start the application**: `./start_desktop_app.sh`
2. **Upload test files**:
   - Small PDF (< 50MB): Should use standard pipeline
   - Large PDF (> 50MB): Should use enhanced pipeline
   - Scanned PDF: Should trigger OCR automatically
3. **Monitor processing**: Watch 7-stage progress in UI
4. **Test chat**: Ask questions about uploaded documents

### **Performance Benchmarking**
```bash
# Monitor system resources during upload
htop  # or Activity Monitor on macOS

# Check processing logs
tail -f backend/app.log

# Measure processing time for your typical files
time curl -X POST "http://localhost:8000/api/upload" -F "files=@your_large_file.pdf"
```

### Getting Help

1. 📖 Check this README thoroughly
2. 🐛 Search existing GitHub issues
3. 💬 Create new issue with detailed info
4. 📧 Contact maintainers for urgent issues

### System Requirements Summary

- **OS**: macOS, Linux, Windows (WSL recommended)
- **Python**: 3.9+ (3.11+ recommended)
- **RAM**: 8GB minimum for Llama3:8b, **16GB+ recommended for large files**
- **Storage**: 10GB for models + documents
- **Network**: Internet for setup only, then fully offline
- **OCR Dependencies**: Tesseract + Poppler (auto-installed via setup script)

---

## 🎉 **Enterprise-Grade OCR-Enabled AI Assistant**

**🦙 Made with ❤️ for privacy-first AI assistance**

*Your personal AI assistant that processes 500MB scanned documents with OCR, respects your privacy, and keeps everything local!*

### **🚀 Key Achievements in This Release:**
- ✅ **500MB file support** (10x increase from 50MB)
- ✅ **OCR for scanned PDFs** with Tesseract integration
- ✅ **100+ language support** for document recognition
- ✅ **Memory-efficient processing** with batch operations
- ✅ **Real-time progress tracking** through 7-stage pipeline
- ✅ **Enhanced architecture** with dual processing pipelines
- ✅ **Complete privacy** - all OCR processing happens locally

**Transform your document workflow from basic text extraction to enterprise-grade document understanding!** 📄🔍🤖 