#!/usr/bin/env python3
"""
Test script for enhanced document service with OCR capabilities
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.document_service import document_service, DocumentService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_document_service():
    """Test the enhanced document service capabilities"""

    logger.info("🧪 Testing Enhanced Document Service")
    logger.info("=" * 50)

    # Test 1: Check capabilities
    logger.info("📋 Checking processing capabilities...")

    from app.services.document_service import (
        IMAGE_AVAILABLE,
        HEIF_AVAILABLE,
        OCR_AVAILABLE,
        PYMUPDF_AVAILABLE,
        PPTX_AVAILABLE,
        EXCEL_AVAILABLE,
    )

    capabilities = {
        "Image Processing (PIL)": IMAGE_AVAILABLE,
        "HEIF Support": HEIF_AVAILABLE,
        "OCR (pytesseract)": OCR_AVAILABLE,
        "Enhanced PDF (PyMuPDF)": PYMUPDF_AVAILABLE,
        "PowerPoint Processing": PPTX_AVAILABLE,
        "Excel Processing": EXCEL_AVAILABLE,
    }

    for capability, available in capabilities.items():
        status = "✅" if available else "❌"
        logger.info(f"   {status} {capability}")

    # Test 2: Check supported file types
    logger.info("\n📁 Checking supported file types...")
    service = DocumentService()

    test_files = [
        "test.pdf",
        "document.docx",
        "presentation.pptx",
        "spreadsheet.xlsx",
        "image.png",
        "photo.jpg",
        "screenshot.jpeg",
        "heic_image.heic",
        "unsupported.txt",
    ]

    for filename in test_files:
        is_supported = service._is_supported_file_type(filename)
        content_type = service._get_content_type(filename)
        status = "✅" if is_supported else "❌"
        logger.info(f"   {status} {filename} -> {content_type}")

    # Test 3: Test file hash generation
    logger.info("\n🔐 Testing file hash generation...")

    test_content = b"This is a test file content for hashing"
    file_hash = service._calculate_file_hash(test_content)
    file_size = len(test_content)
    data_hash = service._calculate_data_hash(test_content, file_size)

    logger.info(f"   File hash: {file_hash[:16]}...")
    logger.info(f"   Data hash: {data_hash[:16]}...")
    logger.info(f"   File size: {file_size} bytes")

    # Test 4: Test new filename generation
    logger.info("\n📝 Testing filename generation...")

    original_filename = "My Document.pdf"
    new_filename = service._generate_new_filename(file_hash, original_filename)
    logger.info(f"   Original: {original_filename}")
    logger.info(f"   New: {new_filename}")

    # Test 5: Check database service
    logger.info("\n🗄️  Testing database service...")

    try:
        files = await document_service.list_documents()
        logger.info(f"   Files in database: {len(files)}")

        if files:
            logger.info("   Recent files:")
            for file in files[:3]:
                logger.info(
                    f"     • {file.get('full_filename', 'Unknown')} ({file.get('content_type', 'Unknown')})"
                )
        else:
            logger.info("   No files in database")

    except Exception as e:
        logger.error(f"   Database error: {e}")

    # Test 6: Test vector service
    logger.info("\n🔍 Testing vector service...")

    try:
        from app.services.vector_service import vector_service

        stats = vector_service.get_collection_stats()
        logger.info(f"   Vector database stats: {stats}")
    except Exception as e:
        logger.error(f"   Vector service error: {e}")

    logger.info("\n✅ Enhanced document service test completed!")


if __name__ == "__main__":
    asyncio.run(test_document_service())
