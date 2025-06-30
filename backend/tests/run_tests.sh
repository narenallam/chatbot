#!/bin/bash

# 🧪 Comprehensive Test Suite Runner with Complete System Cleanup
# Features:
# - Pre-test cleanup for pristine environment
# - Comprehensive test execution with detailed reporting
# - Post-test cleanup
# - Preserves only .env configuration files

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

echo -e "${CYAN}${BOLD}🧪 Comprehensive Test Suite with Complete System Cleanup${NC}"
echo "============================================================="
echo -e "📍 Backend directory: $(pwd)"
echo -e "🗂️  Test suite: Consolidated comprehensive test system"
echo -e "🧹 Cleanup: Complete data reset (preserving .env files only)"
echo -e "🔄 Process: Pre-cleanup → Tests → Post-cleanup"
echo ""

# Check if we're in the backend directory
if [[ ! -d "venv" || ! -f "requirements.txt" ]]; then
    echo -e "${RED}❌ Error: This script must be run from the backend directory${NC}"
    echo "   Usage: cd backend && ./tests/run_tests.sh"
    exit 1
fi

# Default cleanup flags (both enabled by default)
PRE_CLEAN=true
POST_CLEAN=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pre-clean)
            PRE_CLEAN=true
            shift
            ;;
        --no-pre-clean)
            PRE_CLEAN=false
            shift
            ;;
        --post-clean)
            POST_CLEAN=true
            shift
            ;;
        --no-post-clean)
            POST_CLEAN=false
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_tests.sh [options]"
            echo ""
            echo "Options:"
            echo "  --pre-clean      Run pre-test cleanup (default: enabled)"
            echo "  --no-pre-clean   Skip pre-test cleanup"
            echo "  --post-clean     Run post-test cleanup (default: enabled)"
            echo "  --no-post-clean  Skip post-test cleanup"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                    # Run both pre and post cleanup"
            echo "  ./run_tests.sh --no-pre-clean     # Skip pre-cleanup only"
            echo "  ./run_tests.sh --no-post-clean    # Skip post-cleanup only"
            echo "  ./run_tests.sh --no-pre-clean --no-post-clean  # Skip both"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display cleanup configuration
echo -e "${BLUE}🧹 Cleanup Configuration:${NC}"
echo "   Pre-test cleanup: $([ "$PRE_CLEAN" == true ] && echo -e "${GREEN}ENABLED${NC}" || echo -e "${YELLOW}DISABLED${NC}")"
echo "   Post-test cleanup: $([ "$POST_CLEAN" == true ] && echo -e "${GREEN}ENABLED${NC}" || echo -e "${YELLOW}DISABLED${NC}")"
echo ""

# Function to run cleanup
run_cleanup() {
    local cleanup_type="$1"
    echo -e "${PURPLE}🧹 Starting ${cleanup_type} cleanup...${NC}"
    echo "----------------------------------------"
    
    if [ -f "./scripts/cleanup_system.sh" ]; then
        # Use the new independent cleanup script in force mode (non-interactive)
        ./scripts/cleanup_system.sh --force --quiet
        echo -e "${GREEN}✅ ${cleanup_type} cleanup completed${NC}"
    else
        echo -e "${RED}❌ Cleanup script not found: ./scripts/cleanup_system.sh${NC}"
        exit 1
    fi
    echo ""
}

# Function to backup and restore .env files
backup_env_files() {
    echo -e "${BLUE}📋 Backing up .env files...${NC}"
    
    # Create backup directory
    mkdir -p /tmp/chatbot_env_backup_$$
    
    # Backup .env files if they exist
    if [ -f "./.env" ]; then
        cp "./.env" "/tmp/chatbot_env_backup_$$/.env"
        echo "   ✅ Backed up backend/.env"
    fi
    
    if [ -f "../.env" ]; then
        cp "../.env" "/tmp/chatbot_env_backup_$$/../.env"
        echo "   ✅ Backed up root/.env"
    fi
    
    if [ -f "../frontend/.env" ]; then
        mkdir -p "/tmp/chatbot_env_backup_$$/frontend"
        cp "../frontend/.env" "/tmp/chatbot_env_backup_$$/frontend/.env"
        echo "   ✅ Backed up frontend/.env"
    fi
    
    echo "   📂 Backup location: /tmp/chatbot_env_backup_$$"
}

restore_env_files() {
    echo -e "${BLUE}📋 Restoring .env files...${NC}"
    
    # Restore .env files if backups exist
    if [ -f "/tmp/chatbot_env_backup_$$/.env" ]; then
        cp "/tmp/chatbot_env_backup_$$/.env" "./.env"
        echo "   ✅ Restored backend/.env"
    fi
    
    if [ -f "/tmp/chatbot_env_backup_$$/../.env" ]; then
        cp "/tmp/chatbot_env_backup_$$/../.env" "../.env"
        echo "   ✅ Restored root/.env"
    fi
    
    if [ -f "/tmp/chatbot_env_backup_$$/frontend/.env" ]; then
        cp "/tmp/chatbot_env_backup_$$/frontend/.env" "../frontend/.env"
        echo "   ✅ Restored frontend/.env"
    fi
    
    # Clean up backup
    rm -rf "/tmp/chatbot_env_backup_$$"
    echo "   🗑️  Cleaned up backup files"
}

# Cleanup function for script exit
cleanup_on_exit() {
    echo ""
    echo -e "${YELLOW}🛑 Script interrupted or completed${NC}"
    
    # Restore .env files if backup exists
    if [ -d "/tmp/chatbot_env_backup_$$" ]; then
        restore_env_files
    fi
    
    # Run post-test cleanup if enabled
    if [[ "$POST_CLEAN" == true ]]; then
        echo -e "${PURPLE}🧹 Running post-test cleanup...${NC}"
        run_cleanup "POST-TEST"
        
        # Restore .env files after final cleanup
        if [ -d "/tmp/chatbot_env_backup_$$" ]; then
            restore_env_files
        fi
    fi
    
    echo -e "${CYAN}✨ Test execution completed${NC}"
}

# Set up cleanup on script exit
trap cleanup_on_exit EXIT

# PRE-TEST CLEANUP
if [[ "$PRE_CLEAN" == true ]]; then
    echo -e "${BOLD}🚀 PHASE 1: PRE-TEST ENVIRONMENT CLEANUP${NC}"
    echo "=========================================="
    
    # Backup .env files before cleanup
    backup_env_files
    
    # Run complete system reset
    run_cleanup "PRE-TEST"
    
    # Restore .env files after cleanup
    restore_env_files
    
    echo -e "${GREEN}✅ Pre-test cleanup completed - environment is pristine${NC}"
    echo ""
fi

# TEST ENVIRONMENT SETUP
echo -e "${BOLD}🔧 PHASE 2: TEST ENVIRONMENT SETUP${NC}"
echo "==================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating new one...${NC}"
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
    
    # Activate and install requirements
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
    echo "📦 Installing requirements (fresh installation)..."
    pip install -r requirements.txt
else
    # Virtual environment exists, preserve it and optimize package management
    echo "🐍 Activating existing virtual environment..."
    source venv/bin/activate
    
    # Check if pip is functional (but never delete venv)
    echo "📦 Checking virtual environment integrity..."
    if ! python -m pip --version > /dev/null 2>&1; then
        echo -e "${RED}❌ Virtual environment pip is corrupted but preserving venv...${NC}"
        echo "📦 Attempting to repair pip..."
        python3 -m ensurepip --upgrade 2>/dev/null || {
            echo "⚠️  Could not repair pip. Please recreate venv manually if needed."
            echo "📦 Trying to install requirements anyway..."
        }
    fi
    
    # Smart package management - check if requirements are already satisfied
    echo "📦 Checking if requirements are already satisfied..."
    if pip check > /dev/null 2>&1 && pip install -r requirements.txt --dry-run > /dev/null 2>&1; then
        echo "   ✅ All requirements already satisfied, skipping installation"
    else
        echo "📦 Installing/updating missing requirements..."
        pip install -r requirements.txt
    fi
fi

# Check if test data directory exists
if [ ! -d "../test_data" ]; then
    echo -e "${RED}❌ Test data directory not found: ../test_data${NC}"
    echo "   Please ensure test_data folder exists in project root"
    exit 1
fi

# Ensure logs directory exists
mkdir -p logs

echo -e "${GREEN}✅ Environment ready${NC}"
echo ""

# COMPREHENSIVE TEST EXECUTION
echo -e "${BOLD}🧪 PHASE 3: COMPREHENSIVE TEST EXECUTION${NC}"
echo "========================================"
echo ""

# Timestamp for this test run
test_start_time=$(date +"%Y-%m-%d %H:%M:%S")
test_timestamp=$(date +"%Y%m%d_%H%M%S")

echo -e "🚀 Starting comprehensive test execution..."
echo -e "⏰ Start time: $test_start_time"
echo ""

# Run the comprehensive tests
python tests/test_comprehensive_system.py

test_exit_code=$?
test_end_time=$(date +"%Y-%m-%d %H:%M:%S")

echo ""
echo -e "${BOLD}📊 PHASE 4: TEST RESULTS ANALYSIS${NC}"
echo "================================="
echo -e "⏰ End time: $test_end_time"
echo ""

if [ $test_exit_code -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests completed successfully!${NC}"
elif [ $test_exit_code -eq 130 ]; then
    echo -e "${YELLOW}⚠️  Test execution was interrupted by user${NC}"
else
    echo -e "${RED}❌ Some tests failed (exit code: $test_exit_code)${NC}"
fi

# Display detailed results
latest_report=$(ls -t logs/test_report_*.json 2>/dev/null | head -n1)
if [ -n "$latest_report" ]; then
    echo ""
    echo -e "${CYAN}📊 Test Results Summary:${NC}"
    echo "========================"
    
    # Extract key metrics using python if available
    if command -v python >/dev/null 2>&1; then
        python << EOF
import json
import sys
try:
    with open('$latest_report', 'r') as f:
        data = json.load(f)
    
    summary = data.get('test_summary', {})
    print(f"   📋 Total Tests: {summary.get('total_tests', 'N/A')}")
    print(f"   ✅ Passed: {summary.get('passed_tests', 'N/A')}")
    print(f"   ❌ Failed: {summary.get('failed_tests', 'N/A')}")
    print(f"   📈 Success Rate: {summary.get('success_rate', 'N/A')}%")
    print(f"   ⏱️  Duration: {summary.get('total_duration', 'N/A')}s")
    
    # Show any performance metrics
    perf = data.get('performance_benchmarks', {})
    if perf:
        print(f"   🚀 Avg Throughput: {perf.get('average_throughput', 'N/A')} MB/s")
        
except Exception as e:
    print(f"   📄 Report file: $latest_report")
    print(f"   ⚠️  Could not parse JSON: {e}")
EOF
    else
        echo "   📄 Report file: $latest_report"
    fi
    
    echo ""
    echo -e "${BLUE}🎯 View detailed report:${NC}"
    echo "   python scripts/display_test_report.py"
fi

echo ""
echo -e "${BOLD}🎯 SUMMARY${NC}"
echo "=========="
echo -e "   📊 Test execution: $([ $test_exit_code -eq 0 ] && echo -e "${GREEN}SUCCESS${NC}" || echo -e "${RED}FAILED${NC}")"
echo -e "   🧹 Pre-cleanup: $([ "$PRE_CLEAN" == true ] && echo -e "${GREEN}COMPLETED${NC}" || echo -e "${YELLOW}SKIPPED${NC}")"
echo -e "   🧹 Post-cleanup: $([ "$POST_CLEAN" == true ] && echo -e "${GREEN}COMPLETED${NC}" || echo -e "${YELLOW}SKIPPED${NC}")"
echo -e "   📋 .env files: ${GREEN}PRESERVED${NC}"
echo -e "   📄 Test reports: ${GREEN}PRESERVED${NC} for analysis"
echo ""
echo -e "${CYAN}📋 Next Steps:${NC}"
echo -e "   📊 View detailed report: ${BOLD}python scripts/display_test_report.py${NC}"
echo -e "   📁 Latest report: logs/test_report_${test_timestamp}.json"

# Display test report automatically
echo ""
echo -e "${BOLD}📊 DISPLAYING TEST REPORT${NC}"
echo "============================="
if [ -f "scripts/display_test_report.py" ]; then
    python scripts/display_test_report.py
else
    echo -e "${YELLOW}⚠️  display_test_report.py not found in scripts directory${NC}"
    echo -e "${CYAN}📄 Latest report file: $latest_report${NC}"
fi

# Deactivate virtual environment
deactivate

exit $test_exit_code 