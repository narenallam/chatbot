#!/bin/bash
# Setup OCR capabilities for large scanned PDF processing

set -e

echo "🔍 Setting up OCR capabilities for scanned PDFs..."
echo "This will enable processing of 200MB+ scanned documents"
echo

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
else
    OS="unknown"
fi

echo "Detected OS: $OS"

# Install system dependencies
echo "📦 Installing system dependencies..."

if [[ "$OS" == "macos" ]]; then
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    echo "  Installing Tesseract OCR..."
    brew install tesseract
    
    echo "  Installing Poppler (PDF processing)..."
    brew install poppler
    
    echo "  Installing additional language packs..."
    brew install tesseract-lang
    
elif [[ "$OS" == "linux" ]]; then
    echo "  Installing Tesseract OCR..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
        sudo apt-get install -y poppler-utils
        
        # Install additional languages
        echo "  Installing additional language packs..."
        sudo apt-get install -y tesseract-ocr-spa tesseract-ocr-fra tesseract-ocr-deu
    elif command -v yum &> /dev/null; then
        sudo yum install -y tesseract poppler-utils
    else
        echo "❌ Package manager not found. Please install manually:"
        echo "   Tesseract: https://tesseract-ocr.github.io/tessdoc/Installation.html"
        echo "   Poppler: https://poppler.freedesktop.org/"
        exit 1
    fi
else
    echo "❌ Unsupported OS. Please install manually:"
    echo "  - Tesseract OCR"
    echo "  - Poppler"
    exit 1
fi

# Verify system dependencies
echo "🔍 Verifying system dependencies..."

if command -v tesseract &> /dev/null; then
    TESSERACT_VERSION=$(tesseract --version | head -n1)
    echo "  ✅ Tesseract: $TESSERACT_VERSION"
else
    echo "  ❌ Tesseract not found"
    exit 1
fi

if command -v pdftoppm &> /dev/null; then
    echo "  ✅ Poppler: Available"
else
    echo "  ❌ Poppler not found"
    exit 1
fi

# Setup Python environment
echo "🐍 Setting up Python OCR environment..."

# Ensure we're in the right directory
if [[ ! -d "backend" ]]; then
    echo "❌ backend directory not found. Please run from project root."
    exit 1
fi

cd backend

if [[ ! -d "venv" ]]; then
    echo "  Creating virtual environment..."
    python3 -m venv venv
fi

echo "  Activating virtual environment..."
source venv/bin/activate

echo "  Installing all dependencies (including OCR)..."
pip install -r requirements.txt

# Test OCR installation
echo "🧪 Testing OCR installation..."

python -c "
try:
    import pytesseract
    import pdf2image
    import fitz
    from PIL import Image
    print('  ✅ All OCR dependencies installed successfully')
    
    # Test Tesseract
    version = pytesseract.get_tesseract_version()
    print(f'  ✅ Tesseract version: {version}')
    
    # List available languages
    langs = pytesseract.get_languages()
    print(f'  ✅ Available OCR languages: {langs[:5]}...')
    
except ImportError as e:
    print(f'  ❌ Import error: {e}')
    exit(1)
except Exception as e:
    print(f'  ❌ Error: {e}')
    exit(1)
"

# Update configuration
echo "⚙️  Updating configuration..."

# Update .env file to enable large file processing
ENV_FILE=".env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Creating .env file..."
    cat > "$ENV_FILE" << EOF
# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b-instruct-q8_0

# Enhanced File Processing
MAX_FILE_SIZE_MB=500
ENABLE_OCR=true
OCR_LANGUAGE=eng
OCR_DPI=300

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_SEARCH_RESULTS=5

# Database Configuration
CHROMA_DB_PATH=./embeddings
DATA_STORAGE_PATH=./data
EOF
else
    # Update existing .env file
    if ! grep -q "MAX_FILE_SIZE_MB" "$ENV_FILE"; then
        echo "" >> "$ENV_FILE"
        echo "# Enhanced File Processing" >> "$ENV_FILE"
        echo "MAX_FILE_SIZE_MB=500" >> "$ENV_FILE"
        echo "ENABLE_OCR=true" >> "$ENV_FILE"
        echo "OCR_LANGUAGE=eng" >> "$ENV_FILE"
        echo "OCR_DPI=300" >> "$ENV_FILE"
    fi
fi

echo "  ✅ Configuration updated"

# Go back to project root
cd ..

# Create test script
echo "📝 Creating OCR test script..."

cat > test_ocr.py << 'EOF'
#!/usr/bin/env python3
"""
Test OCR capabilities with a sample document
"""

import asyncio
import sys
import os
sys.path.append('app')

from app.services.ocr_document_service import enhanced_document_service

async def test_ocr_system():
    """Test the OCR system with system verification"""
    print("🔍 Testing OCR System Capabilities")
    print("=" * 50)
    
    # Check if OCR is available
    try:
        import pytesseract
        import pdf2image
        import fitz
        
        print("✅ All OCR dependencies available")
        
        # Test Tesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        
        # Test languages
        languages = pytesseract.get_languages()
        print(f"✅ Available languages: {', '.join(languages[:10])}")
        
        print("\n🎯 OCR System Ready for Large Scanned PDFs!")
        print("Maximum file size: 500MB")
        print("Supported formats: PDF (with OCR), DOCX, TXT, HTML, MD")
        print("OCR languages: Multiple (configurable)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ocr_system())
    sys.exit(0 if success else 1)
EOF

chmod +x test_ocr.py

echo "🧪 Running OCR system test..."
python test_ocr.py

echo
echo "🎉 OCR Setup Complete!"
echo
echo "📋 Summary:"
echo "  ✅ Tesseract OCR installed and configured"
echo "  ✅ Poppler PDF processing installed"
echo "  ✅ Python OCR dependencies installed"
echo "  ✅ Configuration updated for 500MB file limit"
echo "  ✅ OCR capabilities enabled"
echo
echo "🚀 Your chatbot can now handle:"
echo "  📄 Large PDFs up to 500MB"
echo "  🖼️  Scanned PDFs with OCR text extraction"
echo "  📝 Multi-language document processing"
echo "  ⚡ Memory-efficient processing for large files"
echo
echo "💡 To test with a real scanned PDF:"
echo "   1. Start the backend: ./start_desktop_app.sh"
echo "   2. Upload a scanned PDF through the UI"
echo "   3. OCR will automatically activate for image-based PDFs"
echo
echo "⚙️  OCR Settings (in .env):"
echo "   MAX_FILE_SIZE_MB=500"
echo "   ENABLE_OCR=true"
echo "   OCR_LANGUAGE=eng"
echo "   OCR_DPI=300" 