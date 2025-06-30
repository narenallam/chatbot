# Backend Tests Directory

This directory contains all test-related files for the Personal Assistant AI Chatbot backend system.

## 📋 Test Organization

All test functionality has been **consolidated** into this directory to eliminate duplication and provide a single source of truth for testing.

### 🧪 Available Test Files

#### 1. **`test_comprehensive_system.py`** - Main Test Suite
**Purpose**: Unified comprehensive test suite combining all testing functionality

```bash
# Run directly
cd backend && python tests/test_comprehensive_system.py

# Or use the test runner (recommended)
cd backend && ./tests/run_tests.sh
```

**Features**:
- 9 comprehensive test cases covering all system components
- Real test data integration from `../test_data/`
- Performance benchmarking with throughput metrics
- Beautiful Rich-formatted console output
- Comprehensive JSON report generation
- Storage system initialization and verification
- SHA256 hash calculation and duplicate detection
- File format conversions (C, JPEG, DOCX, XLSX, PPTX)
- PDF processing (5MB, 19MB, 107MB files)
- OCR functionality on scanned documents
- Parallel processing performance evaluation
- Vector embeddings creation and search operations
- Database operations and session management
- Complete system reset functionality
- Error handling and comprehensive logging

#### 2. **`run_tests.sh`** - Test Runner Script
**Purpose**: Automated test execution with environment setup and reporting

```bash
cd backend && ./tests/run_tests.sh
```

**Features**:
- Automatic virtual environment validation and activation
- Dependency verification and installation
- Test data directory validation
- Environment setup verification
- Automated test execution with error handling
- Quick summary statistics display
- Exit code management for CI/CD integration
- Integration with Rich report viewer

## 📊 Test Coverage

### Core System Tests
- **Storage Initialization**: Directory creation, permissions, statistics
- **Hash Calculation**: SHA256 consistency, duplicate detection
- **File Conversions**: Multi-format support with error handling
- **PDF Processing**: Size-based service selection, throughput measurement
- **Parallel Processing**: Multi-core utilization, performance gains
- **OCR Functionality**: Scanned document processing, text extraction
- **Vector Embeddings**: Document indexing, similarity search
- **Database Operations**: Document storage, session management
- **System Reset**: Complete cleanup, directory recreation

### Performance Benchmarking
- **File Processing Speed**: MB/s throughput measurement
- **Memory Usage**: Resource utilization tracking
- **Service Comparison**: Standard vs Enhanced processing
- **Parallel Efficiency**: Multi-worker performance gains

## 📈 Test Results

### Latest Run Summary
- **Total Tests**: 9
- **Success Rate**: 44.4% (4/9 passed)
- **Duration**: ~14 seconds
- **Performance**: 39.67 MB/s average throughput

### Successful Tests
✅ Storage system initialization  
✅ SHA256 hash calculation and duplicate detection  
✅ PDF processing with large files (107MB)  
✅ OCR functionality with scanned documents

### Failed Tests (Method Signatures)
❌ File format conversions (method compatibility)  
❌ Parallel processing (missing method)  
❌ Vector embeddings (parameter mismatch)  
❌ Database operations (parameter mismatch)  
❌ System reset (missing method)

*Note: Failed tests are due to method signature mismatches and can be fixed by updating the service interfaces.*

## 🔧 Prerequisites

### Required Environment
- **Virtual Environment**: `backend/venv/` with all dependencies installed
- **Test Data**: `../test_data/` directory with sample files
- **System Dependencies**: Tesseract, Poppler (for OCR functionality)

### Test Data Files
The test suite uses real files from `../test_data/`:
- `small.pdf` (5MB) - Standard PDF processing
- `medmium.pdf` (19MB) - Medium PDF processing  
- `large_scanned_file.pdf` (107MB) - OCR and parallel processing
- `narensniff.c` - C source code conversion
- `IMG_CAF37BDF9464-1.jpeg` - Image to PDF conversion
- `All MCA docs.docx` - Word document conversion
- `TrialBal.xlsx` - Excel spreadsheet conversion
- `QM Competitive Programming.pptx` - PowerPoint conversion

## 📄 Reports and Logging

### Test Reports
- **Location**: `backend/logs/test_report_YYYYMMDD_HHMMSS.json`
- **Format**: Comprehensive JSON with all test details and metrics
- **Viewer**: Use `python scripts/display_test_report.py` for beautiful formatted output

### Log Files
- **Test Logs**: `backend/logs/comprehensive_test_results.log`
- **Service Logs**: Individual service logging during test execution
- **Performance Metrics**: Embedded in JSON reports

## 🚀 Usage Examples

### Quick Test Run
```bash
cd backend
./tests/run_tests.sh
```

### Direct Test Execution
```bash
cd backend
source venv/bin/activate
python tests/test_comprehensive_system.py
```

### View Latest Results
```bash
cd backend
python scripts/display_test_report.py
```

### View Specific Report
```bash
cd backend
python scripts/display_test_report.py logs/test_report_20250627_113210.json
```

## 🔄 Integration

### CI/CD Integration
The test runner provides proper exit codes:
- `0`: All tests passed
- `1`: Some tests failed  
- `130`: Interrupted by user

```bash
# In CI/CD pipeline
cd backend && ./tests/run_tests.sh
if [ $? -eq 0 ]; then
    echo "All tests passed"
else
    echo "Tests failed"
    exit 1
fi
```

### Development Workflow
1. **Before commit**: Run `./tests/run_tests.sh`
2. **After changes**: Check `python scripts/display_test_report.py`
3. **Performance validation**: Review throughput metrics
4. **System verification**: Ensure 100% success rate

## 📚 Related Documentation

- **Backend Services**: `../Backend.md`
- **Scripts Documentation**: `../scripts/README.md`
- **Configuration**: `../app/core/config.py`
- **Changelog**: `../../CHANGELOG.md`
- **Main README**: `../../README.md`

## 🔧 Troubleshooting

### Common Issues

1. **Virtual Environment Not Found**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Test Data Missing**
   ```bash
   # Ensure test_data exists in project root
   ls ../test_data/
   ```

3. **Permission Denied**
   ```bash
   chmod +x tests/run_tests.sh
   chmod +x tests/test_comprehensive_system.py
   ```

4. **Import Errors**
   ```bash
   # Check Python path and dependencies
   cd backend
   source venv/bin/activate
   python -c "import app.services.document_service"
   ```

### Method Signature Fixes

To fix the failed tests, update these service methods:
- `vector_service.add_documents()` - Remove `ids` parameter
- `database_service.save_document()` - Remove `content_type` parameter  
- `parallel_pdf_processor.process_pdf_parallel()` - Add missing method
- `enhanced_file_storage.reset_storage_complete()` - Add missing method

## 🎯 Next Steps

1. **Fix Method Signatures**: Update service interfaces for compatibility
2. **Expand Test Coverage**: Add integration tests for API endpoints
3. **Performance Optimization**: Target >50 MB/s throughput
4. **Error Handling**: Improve graceful failure handling
5. **Documentation**: Update service documentation for new methods 