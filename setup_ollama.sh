#!/bin/bash

echo "🦙 Personal Assistant AI Chatbot - Ollama Setup"
echo "==============================================="

# Check if we're on macOS or Linux
OS_TYPE=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="linux"
else
    echo "❌ Unsupported operating system: $OSTYPE"
    echo "This script supports macOS and Linux only."
    exit 1
fi

echo "🖥️  Detected OS: $OS_TYPE"

# Function to install Ollama
install_ollama() {
    echo "📦 Installing Ollama..."
    
    if command -v ollama &> /dev/null; then
        echo "✅ Ollama is already installed"
        ollama --version
        return 0
    fi
    
    if [[ "$OS_TYPE" == "macos" ]]; then
        # macOS installation
        if command -v brew &> /dev/null; then
            echo "🍺 Installing Ollama via Homebrew..."
            brew install ollama
        else
            echo "🌐 Installing Ollama via curl..."
            curl -fsSL https://ollama.ai/install.sh | sh
        fi
    else
        # Linux installation
        echo "🌐 Installing Ollama via curl..."
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
    
    if command -v ollama &> /dev/null; then
        echo "✅ Ollama installed successfully"
        ollama --version
    else
        echo "❌ Failed to install Ollama"
        exit 1
    fi
}

# Function to start Ollama service
start_ollama_service() {
    echo "🚀 Starting Ollama service..."
    
    if [[ "$OS_TYPE" == "macos" ]]; then
        # On macOS, Ollama runs as a regular application
        if ! pgrep -f "ollama serve" > /dev/null; then
            echo "Starting Ollama server..."
            ollama serve &
            OLLAMA_PID=$!
            sleep 3
            echo "✅ Ollama service started (PID: $OLLAMA_PID)"
        else
            echo "✅ Ollama service is already running"
        fi
    else
        # On Linux, check if systemd service exists
        if systemctl is-active --quiet ollama; then
            echo "✅ Ollama service is already running"
        elif systemctl list-unit-files | grep -q ollama; then
            echo "Starting Ollama systemd service..."
            sudo systemctl start ollama
            sudo systemctl enable ollama
            echo "✅ Ollama service started and enabled"
        else
            # Start manually if no systemd service
            if ! pgrep -f "ollama serve" > /dev/null; then
                echo "Starting Ollama server manually..."
                ollama serve &
                OLLAMA_PID=$!
                sleep 3
                echo "✅ Ollama service started (PID: $OLLAMA_PID)"
            else
                echo "✅ Ollama service is already running"
            fi
        fi
    fi
    
    # Wait for service to be ready
    echo "⏳ Waiting for Ollama service to be ready..."
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama service is ready!"
            break
        fi
        echo "   Attempt $i/10..."
        sleep 2
    done
}

# Function to check and verify the required model
check_model() {
    echo "🤖 Checking for required AI model..."
    
    REQUIRED_MODEL="llama3:8b-instruct-q8_0"
    
    # Check if the specific model is available
    if ollama list | grep -q "$REQUIRED_MODEL"; then
        echo "✅ Required model '$REQUIRED_MODEL' is already installed"
        
        # Test the model
        echo "🎯 Testing model..."
        response=$(ollama run "$REQUIRED_MODEL" "Hello, respond with just 'Model working correctly'" 2>/dev/null | head -n 1)
        if [[ -n "$response" ]]; then
            echo "✅ Model test successful: $response"
        else
            echo "⚠️  Model test inconclusive, but model is available"
        fi
    else
        echo "❌ Required model '$REQUIRED_MODEL' not found"
        echo "📋 Available models:"
        ollama list
        echo ""
        echo "🔧 To install the required model, run:"
        echo "   ollama pull $REQUIRED_MODEL"
        echo ""
        read -p "Would you like to install it now? (y/n): " install_choice
        
        if [[ "$install_choice" =~ ^[Yy]$ ]]; then
            echo "📥 Installing $REQUIRED_MODEL..."
            ollama pull "$REQUIRED_MODEL"
            
            if ollama list | grep -q "$REQUIRED_MODEL"; then
                echo "✅ Model installed successfully"
            else
                echo "❌ Failed to install model"
                return 1
            fi
        else
            echo "⚠️  Skipping model installation. The application may not work correctly."
            return 1
        fi
    fi
    
    # Set as default model in config
    echo "⚙️  Configuring application to use $REQUIRED_MODEL..."
    mkdir -p backend
    if [[ ! -f "backend/.env" ]]; then
        cat > backend/.env << EOF
# Ollama Configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=$REQUIRED_MODEL

# Database Configuration
DATABASE_URL=sqlite:///./chatbot.db
CHROMA_DB_PATH=./embeddings
CHROMA_COLLECTION_NAME=documents

# File Storage Configuration
DATA_STORAGE_PATH=./data
MAX_FILE_SIZE_MB=500
ENABLE_OCR=true

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_SEARCH_RESULTS=5
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chat Configuration
MAX_CHAT_HISTORY=10
DEFAULT_TEMPERATURE=0.7

# Application Configuration
DEBUG=false
EOF
    else
        # Update existing .env file
        if grep -q "OLLAMA_MODEL=" backend/.env; then
            sed -i.bak "s/OLLAMA_MODEL=.*/OLLAMA_MODEL=$REQUIRED_MODEL/" backend/.env
        else
            echo "OLLAMA_MODEL=$REQUIRED_MODEL" >> backend/.env
        fi
        echo "Updated OLLAMA_MODEL in backend/.env"
    fi
    
    return 0
}

# Function to test the setup
test_setup() {
    echo "🧪 Testing Ollama setup..."
    
    # Test if Ollama is responding
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "✅ Ollama API is responding"
        
        # List available models
        echo "📋 Available models:"
        ollama list
        
        # Test the specific model
        echo ""
        echo "🎯 Testing llama3:8b-instruct-q8_0 model..."
        echo "Prompt: 'Hello, how are you?'"
        response=$(ollama run llama3:8b-instruct-q8_0 "Hello, how are you?" 2>/dev/null | head -n 1)
        if [[ -n "$response" ]]; then
            echo "✅ Model response: $response"
            echo "🎉 Setup completed successfully!"
        else
            echo "⚠️  Model test failed, but Ollama is running"
        fi
    else
        echo "❌ Ollama API is not responding"
        echo "Please check if Ollama service is running"
        exit 1
    fi
}

# Function to show next steps
show_next_steps() {
    echo ""
    echo "🎯 Next Steps:"
    echo "=============="
    echo "1. Install Python dependencies:"
    echo "   cd backend && source venv/bin/activate && pip install -r requirements.txt"
    echo ""
    echo "2. Start the chatbot application:"
    echo "   ./start_desktop_app.sh"
    echo ""
    echo "3. Open your browser to: http://localhost:8501"
    echo ""
    echo "📝 Configuration:"
    echo "  - Ollama API: http://localhost:11434"
    echo "  - Active model: llama3:8b-instruct-q8_0"
    echo "  - Models installed: $(ollama list | grep -v NAME | wc -l) models"
    echo ""
    echo "🔧 Model management:"
    echo "  - List models: ollama list"
    echo "  - Test model: ollama run llama3:8b-instruct-q8_0 'test message'"
    echo "  - Model info: ollama show llama3:8b-instruct-q8_0"
    echo ""
    echo "💡 Useful commands:"
    echo "  - Check Ollama status: ollama ps"
    echo "  - Stop Ollama: pkill ollama (if running manually)"
    echo "  - Restart Ollama: ollama serve"
}

# Main execution
main() {
    echo "🚀 Starting Ollama setup for Personal Assistant AI Chatbot..."
    echo "🎯 Target model: llama3:8b-instruct-q8_0"
    echo ""
    
    # Install Ollama
    install_ollama
    echo ""
    
    # Start Ollama service
    start_ollama_service
    echo ""
    
    # Check and verify model
    if ! check_model; then
        echo "❌ Model setup failed. Please install the required model manually."
        exit 1
    fi
    echo ""
    
    # Test setup
    test_setup
    echo ""
    
    # Show next steps
    show_next_steps
}

# Run main function
main 