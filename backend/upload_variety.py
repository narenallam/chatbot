#!/usr/bin/env python3
"""
Upload Variety Script - Demonstrates Enhanced Document Processing with OCR and HEIC Support
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.document_service import document_service, DocumentService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_content_type(filename: str) -> str:
    """Get MIME content type from filename using DocumentService helper"""
    return DocumentService()._get_content_type(filename)


async def upload_variety_files():
    """Upload a variety of supported file types to demonstrate enhanced processing with OCR"""

    # Test data directory
    test_data_dir = Path(__file__).parent.parent / "test_data"

    # Files to upload (supported types with OCR focus)
    files_to_upload = [
        # Document files
        "All MCA docs.docx",  # Word document
        "TrialBal.xlsx",  # Excel spreadsheet
        "QM Competitive Programming.pptx",  # PowerPoint
        "Blockchain Portfolio.docx",  # Word document
        "PL of Rossum Products.xlsx",  # Excel
        # PDF files (some may be scanned)
        "small.pdf",  # Small PDF
        "medmium.pdf",  # Medium PDF
        "large_scanned_file.pdf",  # Large PDF (likely scanned)
        "ComputerLabPrograms.pdf",  # PDF
        "Directory of Officers.pdf",  # PDF
        # Image files for OCR testing
        "IMG_7BE1ACB09422-1.jpeg",  # Image
        "Screenshot 2025-06-27 at 7.55.12 am.png",  # Screenshot
        "Screenshot 2025-06-27 at 7.56.00 am.png",  # Screenshot
        # Legacy formats (will be skipped)
        "B+TREE project slides.ppt",  # PowerPoint (legacy)
    ]

    logger.info("🚀 Starting enhanced document processing demonstration...")
    logger.info(f"📁 Test data directory: {test_data_dir}")
    logger.info(f"📊 Files to upload: {len(files_to_upload)}")

    successful_uploads = 0
    failed_uploads = 0
    skipped_uploads = 0
    ocr_used_count = 0

    for filename in files_to_upload:
        file_path = test_data_dir / filename

        if not file_path.exists():
            logger.warning(f"⚠️  File not found: {filename}")
            continue

        try:
            logger.info(
                f"📤 Processing: {filename} ({file_path.stat().st_size / 1024 / 1024:.2f} MB)"
            )

            # Read file content
            with open(file_path, "rb") as f:
                file_content = f.read()

            content_type = get_content_type(filename)

            # Check if file type is supported
            if not DocumentService()._is_supported_file_type(filename):
                logger.warning(f"⚠️  Skipping unsupported file type: {filename}")
                skipped_uploads += 1
                continue

            # Process the file
            result = await document_service.process_uploaded_file(
                file_content=file_content, filename=filename, content_type=content_type
            )

            if result and result.get("status") == "success":
                if result.get("duplicate"):
                    logger.info(f"🔄 Duplicate file detected: {filename}")
                    successful_uploads += 1
                else:
                    successful_uploads += 1
                    logger.info(f"✅ Successfully processed: {filename}")
                    logger.info(f"   📄 Chunks created: {result.get('chunk_count', 0)}")
                    logger.info(
                        f"   ⏱️  Processing time: {result.get('processing_time_seconds', 0):.2f}s"
                    )
                    logger.info(
                        f"   💾 New filename: {result.get('new_filename', 'N/A')}"
                    )

                    # Check if OCR was used
                    if result.get("ocr_used"):
                        ocr_used_count += 1
                        logger.info(f"   🔍 OCR was used for text extraction")
            else:
                failed_uploads += 1
                logger.error(
                    f"❌ Failed to process: {filename} - {result.get('message', '')}"
                )

        except Exception as e:
            failed_uploads += 1
            logger.error(f"❌ Error processing {filename}: {str(e)}")

    # Show final results
    logger.info("\n" + "=" * 60)
    logger.info("📊 PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Successful uploads: {successful_uploads}")
    logger.info(f"❌ Failed uploads: {failed_uploads}")
    logger.info(f"⚠️  Skipped uploads: {skipped_uploads}")
    logger.info(f"🔍 OCR used: {ocr_used_count} files")

    total_processed = successful_uploads + failed_uploads
    if total_processed > 0:
        logger.info(
            f"📈 Success rate: {(successful_uploads / total_processed * 100):.1f}%"
        )

    # Show directory structure
    logger.info("\n📂 STORAGE DIRECTORY STRUCTURE")
    logger.info("=" * 60)

    # Show hashed_files directory
    hashed_files_dir = Path("data/hashed_files")
    if hashed_files_dir.exists():
        files = list(hashed_files_dir.glob("*"))
        logger.info(f"📁 hashed_files/ ({len(files)} files):")
        for file in sorted(files)[:10]:  # Show first 10 files
            size_mb = file.stat().st_size / 1024 / 1024
            logger.info(f"   • {file.name} ({size_mb:.2f} MB)")
        if len(files) > 10:
            logger.info(f"   ... and {len(files) - 10} more files")
    else:
        logger.info("📁 hashed_files/ (directory not found)")

    # Show metadata directory
    metadata_dir = Path("data/metadata")
    if metadata_dir.exists():
        files = list(metadata_dir.glob("*"))
        logger.info(f"📁 metadata/ ({len(files)} files):")
        for file in sorted(files)[:5]:  # Show first 5 files
            logger.info(f"   • {file.name}")
        if len(files) > 5:
            logger.info(f"   ... and {len(files) - 5} more files")
    else:
        logger.info("📁 metadata/ (directory not found)")

    # Show database information
    logger.info("\n🗄️  DATABASE INFORMATION")
    logger.info("=" * 60)

    try:
        files = await document_service.list_documents()
        logger.info(f"📊 Total files in database: {len(files)}")

        # Group by content type
        content_types = {}
        ocr_files = []
        for file in files:
            content_type = file.get("content_type", "unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1

            # Check for OCR usage
            if file.get("metadata", {}).get("ocr_used"):
                ocr_files.append(file.get("full_filename", "Unknown"))

        logger.info("📋 Files by type:")
        for content_type, count in content_types.items():
            logger.info(f"   • {content_type}: {count} files")

        if ocr_files:
            logger.info(f"🔍 Files processed with OCR ({len(ocr_files)}):")
            for filename in ocr_files[:5]:  # Show first 5
                logger.info(f"   • {filename}")
            if len(ocr_files) > 5:
                logger.info(f"   ... and {len(ocr_files) - 5} more")

    except Exception as e:
        logger.error(f"❌ Error accessing database: {e}")

    # Show capabilities
    logger.info("\n🔧 PROCESSING CAPABILITIES")
    logger.info("=" * 60)

    # Check what's available
    from app.services.document_service import (
        IMAGE_AVAILABLE,
        HEIF_AVAILABLE,
        OCR_AVAILABLE,
        PYMUPDF_AVAILABLE,
        PPTX_AVAILABLE,
        EXCEL_AVAILABLE,
    )

    capabilities = []
    if IMAGE_AVAILABLE:
        capabilities.append("✅ Image processing (PIL)")
    else:
        capabilities.append("❌ Image processing (PIL)")

    if HEIF_AVAILABLE:
        capabilities.append("✅ HEIF support")
    else:
        capabilities.append("❌ HEIF support")

    if OCR_AVAILABLE:
        capabilities.append("✅ OCR (pytesseract)")
    else:
        capabilities.append("❌ OCR (pytesseract)")

    if PYMUPDF_AVAILABLE:
        capabilities.append("✅ Enhanced PDF (PyMuPDF)")
    else:
        capabilities.append("❌ Enhanced PDF (PyMuPDF)")

    if PPTX_AVAILABLE:
        capabilities.append("✅ PowerPoint processing")
    else:
        capabilities.append("❌ PowerPoint processing")

    if EXCEL_AVAILABLE:
        capabilities.append("✅ Excel processing")
    else:
        capabilities.append("❌ Excel processing")

    for capability in capabilities:
        logger.info(f"   {capability}")

    logger.info("\n🎉 Enhanced document processing demonstration completed!")
    logger.info("💡 Check the data/hashed_files/ directory to see the stored files")
    logger.info("💡 Check the database for file metadata and chunk information")
    logger.info("🔍 OCR was used for images and scanned documents")


if __name__ == "__main__":
    asyncio.run(upload_variety_files())
