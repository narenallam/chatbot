#!/bin/bash

# 🧹 Independent System Cleanup Script for Personal Assistant AI Chatbot
# Purpose: Complete data cleanup including databases while preserving critical infrastructure
# Usage: ./scripts/cleanup_system.sh [--force] [--skip-backup]
# 
# What this script does:
# ✅ Clears all databases (vector, SQLite)
# ✅ Removes all uploaded files and processed data
# ✅ Clears all logs and temporary files
# ✅ Resets storage directories
# ✅ Preserves .env files (critical configuration)
# ✅ Preserves virtual environment (never deleted)
# ✅ Preserves test reports for analysis

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Parse command line arguments
FORCE_CLEANUP=false
SKIP_BACKUP=false
QUIET_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE_CLEANUP=true
            shift
            ;;
        --skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        --quiet|-q)
            QUIET_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--force] [--skip-backup] [--quiet]"
            exit 1
            ;;
    esac
done

# Helper function for safe file/directory removal
safe_remove_file() {
    local file_path="$1"
    local description="$2"
    
    if [ -f "$file_path" ]; then
        if [[ "$QUIET_MODE" == false ]]; then
            echo "   🗑️  Removing $description: $file_path"
        fi
        rm -f "$file_path" 2>/dev/null || {
            echo "   ⚠️  Could not remove $file_path"
        }
    fi
}

safe_remove_dir() {
    local dir_path="$1"
    local description="$2"
    
    if [ -d "$dir_path" ]; then
        if [[ "$QUIET_MODE" == false ]]; then
            echo "   🗑️  Removing $description: $dir_path"
        fi
        rm -rf "$dir_path" 2>/dev/null || {
            echo "   ⚠️  Could not remove $dir_path"
        }
    fi
}

# Ensure we're in the backend directory
if [[ ! -d "venv" && ! -f "requirements.txt" ]]; then
    echo -e "${RED}❌ Error: This script must be run from the backend directory${NC}"
    echo "   Usage: cd backend && ./scripts/cleanup_system.sh"
    exit 1
fi

echo -e "${CYAN}${BOLD}🧹 Independent System Cleanup Script${NC}"
echo "==========================================="
echo -e "📍 Backend directory: $(pwd)"
echo -e "🎯 Purpose: Complete data cleanup while preserving infrastructure"
echo ""

# Show what will be preserved vs cleaned
echo -e "${GREEN}${BOLD}✅ PRESERVED (Never Deleted):${NC}"
echo "   • Virtual environment (venv/)"
echo "   • Environment files (.env)"
echo "   • Test reports (*.json)"
echo "   • Core application code"
echo ""

echo -e "${YELLOW}${BOLD}🗑️  WILL BE CLEANED:${NC}"
echo "   • Vector database (ChromaDB)"
echo "   • SQLite database"
echo "   • All uploaded files"
echo "   • All processed files"
echo "   • All logs and temporary files"
echo "   • Python cache files"
echo "   • Storage directories"
echo ""

# Confirmation (unless force mode)
if [[ "$FORCE_CLEANUP" == false ]]; then
    echo -e "${RED}${BOLD}⚠️  WARNING: This will permanently delete all data!${NC}"
    read -p "Do you want to continue? (y/N): " confirm
    
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "Cleanup cancelled."
        exit 0
    fi
fi

echo -e "${BLUE}🚀 Starting system cleanup...${NC}"
echo ""

# Create backup of critical files if not skipped
if [[ "$SKIP_BACKUP" == false ]]; then
    echo "💾 Creating backup of critical files..."
    mkdir -p ./cleanup_backup 2>/dev/null || true
    
    # Backup .env files
    if [ -f "./.env" ]; then
        cp ./.env ./cleanup_backup/.env.backup 2>/dev/null || true
        echo "   ✅ Backed up .env file"
    fi
    
    # Backup test reports
    if [ -d "./logs" ]; then
        find ./logs -name "test_report_*.json" -exec cp {} ./cleanup_backup/ \; 2>/dev/null || true
        preserved_reports=$(find ./cleanup_backup -name "test_report_*.json" 2>/dev/null | wc -l)
        if [ "$preserved_reports" -gt 0 ]; then
            echo "   ✅ Backed up $preserved_reports test reports"
        fi
    fi
    echo ""
fi

# 1. Clear Vector Database
echo "🗂️  Clearing Vector Database..."
safe_remove_dir "./chroma_db" "ChromaDB vector database"
safe_remove_dir "./vector_storage" "Vector storage directory"
safe_remove_dir "./embeddings" "Embeddings cache"
echo ""

# 2. Clear SQLite Database
echo "🗄️  Clearing SQLite Database..."
safe_remove_file "./chatbot.db" "Main SQLite database"
safe_remove_file "./database.db" "Alternative database file"
safe_remove_file "./app.db" "App database file"
safe_remove_file "./*.db" "Any other database files"
echo ""

# 3. Clear File Storage
echo "📁 Clearing File Storage..."
safe_remove_dir "./data" "Data directory"
safe_remove_dir "./uploads" "Uploads directory"
safe_remove_dir "./storage" "Storage directory"
safe_remove_dir "./temp" "Temporary files directory"
safe_remove_dir "./converted_files" "Converted files directory"
safe_remove_dir "./original_files" "Original files directory"
safe_remove_dir "./processed_files" "Processed files directory"
echo ""

# 4. Clear Logs (preserve test reports)
echo "📝 Clearing Log Files..."
echo "🗑️  Preserving test reports and cleaning other logs..."

# Preserve test reports before removing logs directory
mkdir -p ./logs_backup 2>/dev/null || true
if [ -d "./logs" ]; then
    find ./logs -name "test_report_*.json" -exec cp {} ./logs_backup/ \; 2>/dev/null || true
    preserved_count=$(find ./logs_backup -name "test_report_*.json" 2>/dev/null | wc -l)
    if [ "$preserved_count" -gt 0 ]; then
        echo "   💾 Preserved $preserved_count test reports"
    fi
fi

# Remove logs directory
safe_remove_dir "./logs" "Logs directory"
safe_remove_dir "../logs" "Root logs directory"
safe_remove_file "./chatbot.log" "Backend log file"
safe_remove_file "../chatbot.log" "Root log file"
safe_remove_file "./error.log" "Backend error log"
safe_remove_file "./access.log" "Backend access log"

# Restore test reports
if [ -d "./logs_backup" ] && [ "$(find ./logs_backup -name 'test_report_*.json' 2>/dev/null | wc -l)" -gt 0 ]; then
    mkdir -p ./logs
    cp ./logs_backup/test_report_*.json ./logs/ 2>/dev/null || true
    rm -rf ./logs_backup
    restored_count=$(find ./logs -name "test_report_*.json" 2>/dev/null | wc -l)
    echo "   ✅ Restored $restored_count test reports"
fi
echo ""

# 5. Clear Cache and Temporary Files
echo "🧹 Clearing Cache Files..."
safe_remove_dir "./__pycache__" "Backend Python cache"
safe_remove_dir "./app/__pycache__" "App Python cache"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true
find . -type f -name ".DS_Store" -delete 2>/dev/null || true
echo ""

# 6. Clear Test Artifacts (preserve test reports)
echo "🧪 Clearing Test Artifacts..."
safe_remove_dir "./test_results" "Backend test results"
safe_remove_dir "./coverage" "Backend coverage reports"
safe_remove_dir "../test_results" "Root test results"
safe_remove_file "./pytest.ini" "Pytest configuration"
safe_remove_file "./.coverage" "Coverage data file"

# Remove any test-generated files (but preserve test reports)
safe_remove_file "./test_*.log" "Test log files"
safe_remove_file "./comprehensive_test_*.log" "Comprehensive test logs"
find ./logs -name "system_reset_*.log" -delete 2>/dev/null || true
echo "   ✅ Test log files cleaned"
echo "   ℹ️  Test reports (test_report_*.json) PRESERVED for analysis"
echo ""

# 7. Clean Virtual Environment Cache (NEVER delete venv)
echo "🐍 Cleaning Virtual Environment Cache..."
if [ -d "./venv" ]; then
    echo "🧹 Cleaning only cache files (preserving virtual environment)..."
    
    # Only clean cache and temporary files, NEVER delete the venv itself
    find venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find venv -type f -name "*.pyc" -delete 2>/dev/null || true
    find venv -type f -name "*.pyo" -delete 2>/dev/null || true
    
    # Clean pip cache directories safely without touching pip installation
    find venv -type d -path "*/.cache" -exec rm -rf {} + 2>/dev/null || true
    find venv -type d -name "pip-build-*" -exec rm -rf {} + 2>/dev/null || true
    find venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    
    echo "   ✅ Virtual environment cache cleaned (venv structure preserved)"
    echo "   ℹ️  Virtual environment ready for reuse"
else
    echo "   ℹ️  Virtual environment not found (will be created when needed)"
fi
echo ""

# 8. Recreate Essential Directory Structure
echo "📁 Recreating Storage Directory Structure..."
mkdir -p "./data" 2>/dev/null || true
mkdir -p "./data/hashed_files" 2>/dev/null || true
mkdir -p "./data/original_files" 2>/dev/null || true
mkdir -p "./data/metadata" 2>/dev/null || true
mkdir -p "./data/temp" 2>/dev/null || true
mkdir -p "./logs" 2>/dev/null || true
mkdir -p "./embeddings" 2>/dev/null || true
echo "   ✅ Essential directories recreated"
echo ""

# 9. Restore critical files from backup
if [[ "$SKIP_BACKUP" == false ]] && [ -d "./cleanup_backup" ]; then
    echo "🔄 Restoring critical files..."
    
    # Restore .env files
    if [ -f "./cleanup_backup/.env.backup" ]; then
        cp ./cleanup_backup/.env.backup ./.env 2>/dev/null || true
        echo "   ✅ Restored .env file"
    fi
    
    # Clean up backup
    rm -rf ./cleanup_backup 2>/dev/null || true
    echo ""
fi

# 10. Verification
echo -e "${PURPLE}🔍 Verification...${NC}"
echo "Checking critical preservation:"

# Check .env files
if [ -f "./.env" ]; then
    echo "   ✅ Environment file (.env) preserved"
else
    echo "   ⚠️  Environment file (.env) not found"
fi

# Check virtual environment
if [ -d "./venv" ]; then
    echo "   ✅ Virtual environment preserved and ready for reuse"
else
    echo "   ℹ️  Virtual environment not found (will be created when needed)"
fi

# Check test reports
test_reports=$(find ./logs -name "test_report_*.json" 2>/dev/null | wc -l)
if [ "$test_reports" -gt 0 ]; then
    echo "   ✅ Test reports preserved ($test_reports files)"
else
    echo "   ℹ️  No test reports found"
fi

# Check databases are cleared
if [ ! -f "./chatbot.db" ] && [ ! -d "./chroma_db" ]; then
    echo "   ✅ Databases successfully cleared"
else
    echo "   ⚠️  Some database files may still exist"
fi

echo ""

# Final Summary
echo -e "${GREEN}${BOLD}✅ CLEANUP COMPLETED SUCCESSFULLY!${NC}"
echo ""
echo "📊 Cleanup Summary:"
echo "   • Vector database: CLEARED"
echo "   • SQLite database: CLEARED"  
echo "   • Original files: CLEARED"
echo "   • Converted files: CLEARED"
echo "   • Metadata: CLEARED"
echo "   • Logs: CLEARED"
echo "   • Test artifacts: CLEARED"
echo "   • Python cache: CLEARED"
echo "   • Storage structure: RECREATED"
echo "   • Environment files (.env): ALWAYS PRESERVED"
echo "   • Test reports (*.json): ALWAYS PRESERVED"
echo "   • Virtual environment (venv): NEVER DELETED"
echo ""

echo -e "${BLUE}🚀 Next Steps:${NC}"
echo "   1. System is ready for fresh data"
echo "   2. Virtual environment is preserved and ready"
echo "   3. Configuration files (.env) are intact"
echo "   4. You can start uploading new documents"
echo ""

if [[ "$QUIET_MODE" == false ]]; then
    echo -e "${CYAN}💡 Usage Examples:${NC}"
    echo "   • Interactive cleanup: ./scripts/cleanup_system.sh"
    echo "   • Force cleanup: ./scripts/cleanup_system.sh --force"
    echo "   • Quiet cleanup: ./scripts/cleanup_system.sh --quiet"
    echo "   • Skip backup: ./scripts/cleanup_system.sh --skip-backup"
fi

echo -e "${GREEN}🎉 System cleanup completed! Ready for fresh start.${NC}" 