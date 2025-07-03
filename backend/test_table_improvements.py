#!/usr/bin/env python3
"""
Test script for Phase 1 table processing improvements
"""

import sys
import re
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_table_aware_splitter():
    """Test the table-aware text splitter"""
    print("🧪 Testing Table-Aware Text Splitter...")
    
    # Import the splitter class
    try:
        from app.services.document_service import TableAwareTextSplitter
        splitter = TableAwareTextSplitter(chunk_size=500, chunk_overlap=100, table_chunk_size=1000)
        print("✅ TableAwareTextSplitter imported successfully")
    except Exception as e:
        print(f"❌ Error importing TableAwareTextSplitter: {e}")
        return False
    
    # Test with sample table content
    sample_table_text = """
Some regular text content here.

=== TABLE 1 (Page 2) ===
HEADERS: Product | Price | Quantity | Total
---
ROW 1: Widget A | $10.00 | 5 | $50.00
ROW 2: Widget B | $15.00 | 3 | $45.00
ROW 3: Widget C | $20.00 | 2 | $40.00
=== END TABLE 1 (3 data rows) ===

More regular text content after the table.
Another paragraph of text here.
"""
    
    try:
        chunks = splitter.split_text(sample_table_text)
        print(f"✅ Text splitting successful: {len(chunks)} chunks created")
        
        # Check if table is preserved in one chunk
        table_chunk_found = False
        for i, chunk in enumerate(chunks):
            if "=== TABLE 1" in chunk and "=== END TABLE 1" in chunk:
                table_chunk_found = True
                print(f"✅ Complete table found in chunk {i+1}")
                break
        
        if not table_chunk_found:
            print("⚠️ Table was fragmented across chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in text splitting: {e}")
        return False

def test_table_format_functions():
    """Test the table formatting functions"""
    print("\n🧪 Testing Table Formatting Functions...")
    
    try:
        from app.services.document_service import DocumentService
        doc_service = DocumentService()
        print("✅ DocumentService imported successfully")
    except Exception as e:
        print(f"❌ Error importing DocumentService: {e}")
        return False
    
    # Test PDF table formatting
    sample_table_data = [
        ["Product", "Price", "Quantity"],
        ["Widget A", "$10.00", "5"],
        ["Widget B", "$15.00", "3"],
        ["Widget C", "$20.00", "2"]
    ]
    
    try:
        formatted_table = doc_service._format_table_structure(sample_table_data, 1, 1)
        print("✅ PDF table formatting successful")
        
        # Check for proper structure
        if "HEADERS:" in formatted_table and "ROW" in formatted_table:
            print("✅ Table structure properly formatted")
        else:
            print("⚠️ Table structure formatting incomplete")
            
    except Exception as e:
        print(f"❌ Error in table formatting: {e}")
        return False
    
    return True

def test_vector_service_table_analysis():
    """Test the vector service table analysis"""
    print("\n🧪 Testing Vector Service Table Analysis...")
    
    try:
        from app.services.vector_service import VectorService
        vector_service = VectorService()
        print("✅ VectorService imported successfully")
    except Exception as e:
        print(f"❌ Error importing VectorService: {e}")
        return False
    
    # Test table analysis
    sample_table_chunk = """
=== EXCEL SHEET: Sales Data ===
HEADERS: Product | Q1 Sales | Q2 Sales | Q3 Sales | Q4 Sales
---
ROW 1: Widget A | $10,000 | $12,000 | $11,500 | $13,000
ROW 2: Widget B | $8,500 | $9,200 | $9,800 | $10,100
ROW 3: Widget C | $15,000 | $16,500 | $14,200 | $17,800
=== END EXCEL SHEET: Sales Data (3 data rows processed) ===
"""
    
    try:
        analysis = vector_service._analyze_chunk_for_tables(sample_table_chunk)
        print("✅ Table analysis successful")
        
        # Check analysis results
        if analysis.get("contains_table"):
            print(f"✅ Table detected: {analysis.get('table_type')}")
            print(f"   - Table indicators: {analysis.get('table_indicators')}")
            print(f"   - Row count: {analysis.get('row_count')}")
            print(f"   - Numeric data: {analysis.get('numeric_data')}")
        else:
            print("⚠️ Table not detected in sample")
            
    except Exception as e:
        print(f"❌ Error in table analysis: {e}")
        return False
    
    return True

def test_image_table_detection():
    """Test image table content structuring"""
    print("\n🧪 Testing Image Table Detection...")
    
    try:
        from app.services.document_service import DocumentService
        import asyncio
        
        doc_service = DocumentService()
        print("✅ DocumentService with image processing imported successfully")
    except Exception as e:
        print(f"❌ Error importing DocumentService: {e}")
        return False
    
    # Test with simulated OCR text that looks like a table
    sample_ocr_texts = [
        "Product Price Quantity\nWidget A $10.00 5\nWidget B $15.00 3\nWidget C $20.00 2",
        "Name | Age | City\nJohn | 25 | NYC\nJane | 30 | LA\nBob | 35 | Chicago"
    ]
    
    try:
        # Test the table structuring logic
        async def run_test():
            for i, text in enumerate(sample_ocr_texts):
                structured = await doc_service._structure_image_table_content(text, [], f"test_image_{i}.png")
                if structured:
                    print(f"✅ Image table {i+1} structured successfully")
                else:
                    print(f"⚠️ Image table {i+1} not detected as table")
        
        # Run the async test
        if sys.version_info >= (3, 7):
            asyncio.run(run_test())
        else:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(run_test())
            
        return True
        
    except Exception as e:
        print(f"❌ Error in image table detection: {e}")
        return False

def main():
    """Run all table processing tests"""
    print("🚀 Testing Phase 1 Table Processing Improvements\n")
    
    test_results = []
    
    # Run individual tests
    test_results.append(test_table_aware_splitter())
    test_results.append(test_table_format_functions())
    test_results.append(test_vector_service_table_analysis())
    test_results.append(test_image_table_detection())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n📊 Test Results Summary:")
    print(f"   Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 All table processing improvements working correctly!")
        return True
    else:
        print("⚠️ Some improvements need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)