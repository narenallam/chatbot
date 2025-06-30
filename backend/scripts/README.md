# Backend Scripts

This directory contains all backend-related scripts for the Personal Assistant AI Chatbot.

## 📋 Available Scripts

### 🧹 Independent System Cleanup ✨ **NEW**
**File**: `cleanup_system.sh`  
**Purpose**: Complete data cleanup including databases while preserving critical infrastructure.

```bash
cd backend && ./scripts/cleanup_system.sh                    # Interactive mode
cd backend && ./scripts/cleanup_system.sh --force            # Non-interactive mode
cd backend && ./scripts/cleanup_system.sh --force --quiet    # Silent cleanup
```

**What it does**:
- ✅ Clears all databases (vector, SQLite)
- ✅ Removes all uploaded files and processed data
- ✅ Clears all logs and temporary files
- ✅ Resets storage directories
- ✅ **Preserves .env files** (critical configuration)
- ✅ **Preserves virtual environment** (never deleted)
- ✅ **Preserves test reports** for analysis

**Command Options**:
- `--force` or `-f`: Skip confirmation dialog
- `--quiet` or `-q`: Minimal output
- `--skip-backup`: Skip backing up files

**⚠️ Warning**: This will permanently delete ALL data except preserved items. Use with caution.

### 🗑️ Complete System Reset (Legacy)
**File**: `reset_system_complete.sh`  
**Purpose**: ⚠️ **Legacy script - use `cleanup_system.sh` instead**

```bash
cd backend && ./scripts/reset_system_complete.sh
```

**What it does**:
- Stops all running services (backend, frontend, Ollama)
- Removes vector database (ChromaDB embeddings)
- Removes SQLite database (conversations, documents, sessions)
- Removes all file storage (originals, converted, temp, metadata)
- Removes log files and Python cache
- Recreates essential directory structure
- Preserves virtual environment
- Creates detailed reset log

**⚠️ Warning**: This will permanently delete ALL data. Use with extreme caution.

### 🎨 Beautiful Test Report Viewer
**File**: `display_test_report.py`  
**Purpose**: Beautiful terminal-based test report viewer using Rich module.

```bash
cd backend && python scripts/display_test_report.py [report_file]
```

**Features**:
- Automatically finds latest test report if no file specified
- Color-coded test results with status indicators
- Performance benchmark tables with throughput metrics
- Detailed test breakdown in hierarchical tree view
- System information and comprehensive statistics
- Beautiful Rich-formatted panels and tables

## 🧪 Test Organization

**All tests have been consolidated in the `tests/` directory:**

### 📋 Comprehensive Test Suite
**File**: `tests/test_comprehensive_system.py`  
**Purpose**: Unified test suite combining all testing functionality.

```bash
cd backend && python tests/test_comprehensive_system.py
```

**Test Coverage**:
- Storage system initialization and organization
- SHA256 hash calculation and duplicate detection
- File format conversions (C source, JPEG, DOCX, XLSX, PPTX)
- PDF processing (5MB, 19MB, 107MB files)
- OCR functionality on scanned documents
- Parallel processing performance evaluation
- Vector embeddings creation and search operations
- Database operations and session management
- Complete system reset functionality
- Error handling and comprehensive logging

### 🚀 Test Runner Script
**File**: `tests/run_tests.sh`  
**Purpose**: Simplified test execution with automatic environment setup.

```bash
cd backend && ./tests/run_tests.sh
```

**What it does**:
- Validates backend directory and virtual environment
- Activates virtual environment automatically
- Checks and installs missing dependencies
- Runs the consolidated comprehensive test suite
- Displays quick summary statistics
- Provides clear success/failure feedback with next steps
- Automatically deactivates virtual environment

## 📁 Directory Requirements

All scripts must be run from the `backend` directory:

```bash
# Correct usage
cd backend
./scripts/reset_system_complete.sh        # System reset
python scripts/display_test_report.py     # View test reports
./tests/run_tests.sh                      # Run all tests
python tests/test_comprehensive_system.py # Run tests directly

# Incorrect - will fail
./backend/scripts/reset_system_complete.sh  # Don't run from project root
```

## 🔧 Prerequisites

### Virtual Environment
- Scripts assume virtual environment is located at `backend/venv/`
- Virtual environment should contain all required dependencies from `requirements.txt`

### Test Data
- Test scripts require `test_data/` directory in project root
- Test data should include sample files of various formats and sizes

### System Dependencies
- **OCR Testing**: Requires Tesseract and Poppler (optional)
- **Vector Database**: ChromaDB for embeddings testing
- **Database**: SQLite for metadata testing

## 📊 Script Exit Codes

| Exit Code | Meaning |
|-----------|---------|
| 0 | Success |
| 1 | General error |
| 130 | Interrupted by user (Ctrl+C) |

## 📝 Logging

### Test Runner Logs
- **Location**: `backend/logs/test_report_YYYYMMDD_HHMMSS.json`
- **Format**: JSON with detailed test results and performance metrics
- **Retention**: Manual cleanup required

### Reset Logs  
- **Location**: `backend/logs/system_reset_YYYYMMDD_HHMMSS.log`
- **Format**: Plain text with step-by-step reset details
- **Retention**: Manual cleanup required

## 🛠️ Development

### Adding New Scripts
1. Create script in `backend/scripts/` directory
2. Make executable: `chmod +x script_name.sh`
3. Add validation for backend directory location
4. Update this README with usage instructions

### Script Best Practices
- Always validate we're in the correct directory
- Use proper error handling (`set -e` for bash scripts)
- Provide clear user feedback and progress indicators
- Include usage instructions in script comments
- Test scripts on clean environment before committing

## 🔗 Related Documentation

- **Backend Services**: `backend/Backend.md`
- **Consolidated Test Suite**: `backend/tests/test_comprehensive_system.py`
- **Test Runner**: `backend/tests/run_tests.sh`
- **Configuration**: `backend/app/core/config.py`
- **Changelog**: `CHANGELOG.md`
- **Main README**: `README.md` 