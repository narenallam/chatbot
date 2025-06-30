#!/usr/bin/env python3
"""
Upload Variety Script - Demonstrates Enhanced File Storage with Various File Types
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.document_service import DocumentService
from app.services.enhanced_file_storage import EnhancedFileStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def upload_variety_files():
    """Upload a variety of file types to demonstrate enhanced storage"""

    # Initialize services
    storage = EnhancedFileStorage()
    doc_service = DocumentService()

    # Test data directory
    test_data_dir = Path(__file__).parent.parent / "test_data"

    # Files to upload (various types)
    files_to_upload = [
        "narensniff.c",  # C code file
        "IMG_CAF37BDF9464-1.jpeg",  # Image file
        "All MCA docs.docx",  # Word document
        "TrialBal.xlsx",  # Excel spreadsheet
        "QM Competitive Programming.pptx",  # PowerPoint
        "README.md",  # Markdown file
        "small.pdf",  # Small PDF
        "medmium.pdf",  # Medium PDF
        "large_scanned_file.pdf",  # Large PDF
        "B+TREE project slides.ppt",  # PowerPoint
        "Blockchain Portfolio.docx",  # Word document
        "ComputerLabPrograms.pdf",  # PDF
        "Directory of Officers.pdf",  # PDF
        "Financials Rossum AY 2019-20.xlsx",  # Excel
        "IMG_7BE1ACB09422-1.jpeg",  # Image
        "NMM socket Task.doc",  # Word document
        "nping.c",  # C code
        "ntracer_icmp.c",  # C code
        "Screenshot 2025-06-27 at 7.55.12 am.png",  # Screenshot
        "Screenshot 2025-06-27 at 7.56.00 am.png",  # Screenshot
        "setmtu.c",  # C code
        "PL of Rossum Products.xlsx",  # Excel
    ]

    logger.info("🚀 Starting variety file upload demonstration...")
    logger.info(f"📁 Test data directory: {test_data_dir}")
    logger.info(f"📊 Files to upload: {len(files_to_upload)}")

    successful_uploads = 0
    failed_uploads = 0

    for filename in files_to_upload:
        file_path = test_data_dir / filename

        if not file_path.exists():
            logger.warning(f"⚠️  File not found: {filename}")
            continue

        try:
            logger.info(
                f"📤 Uploading: {filename} ({file_path.stat().st_size / 1024 / 1024:.2f} MB)"
            )

            # Process the file
            result = await doc_service.process_document(str(file_path))

            if result and result.get("status") == "success":
                successful_uploads += 1
                logger.info(f"✅ Successfully processed: {filename}")
                logger.info(f"   📄 Chunks created: {result.get('chunk_count', 0)}")
                logger.info(
                    f"   ⏱️  Processing time: {result.get('processing_time', 0):.2f}s"
                )
            else:
                failed_uploads += 1
                logger.error(f"❌ Failed to process: {filename}")

        except Exception as e:
            failed_uploads += 1
            logger.error(f"❌ Error processing {filename}: {str(e)}")

    # Show final results
    logger.info("\n" + "=" * 60)
    logger.info("📊 UPLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Successful uploads: {successful_uploads}")
    logger.info(f"❌ Failed uploads: {failed_uploads}")
    logger.info(
        f"📈 Success rate: {(successful_uploads / (successful_uploads + failed_uploads) * 100):.1f}%"
    )

    # Show storage statistics
    stats = storage.get_storage_stats()
    logger.info("\n📁 STORAGE STATISTICS")
    logger.info("=" * 60)
    logger.info(
        f"📂 Original files: {stats['original_files']['count']} files, {stats['original_files']['size_mb']:.2f} MB"
    )
    logger.info(
        f"📄 Converted files: {stats['converted_files']['count']} files, {stats['converted_files']['size_mb']:.2f} MB"
    )
    logger.info(
        f"📋 Metadata files: {stats.get('metadata_files', {}).get('count', 0)} files"
    )
    logger.info(f"📊 Total files: {stats['total_files']}")

    # Show directory structure
    logger.info("\n📂 DIRECTORY STRUCTURE")
    logger.info("=" * 60)

    for subdir in ["originals", "converted", "metadata"]:
        dir_path = Path("data") / subdir
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            logger.info(f"📁 {subdir}/ ({len(files)} files):")
            for file in sorted(files)[:10]:  # Show first 10 files
                logger.info(f"   • {file.name}")
            if len(files) > 10:
                logger.info(f"   ... and {len(files) - 10} more files")
        else:
            logger.info(f"📁 {subdir}/ (directory not found)")

    logger.info("\n🎉 Variety upload demonstration completed!")
    logger.info("💡 Check the data/ directory to see the organized file structure")


if __name__ == "__main__":
    asyncio.run(upload_variety_files())
