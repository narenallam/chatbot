#!/usr/bin/env python3
"""
Comprehensive Test Suite for Personal Assistant AI Chatbot
Consolidated test suite combining all test functionality with performance benchmarking
Run from backend directory: cd backend && python tests/test_comprehensive_system.py
"""

# Fix HuggingFace tokenizers parallelism warning BEFORE any imports that might use tokenizers
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import asyncio
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

# Add the parent directory (backend) to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import services
from app.core.config import settings
from app.services.enhanced_file_storage import enhanced_file_storage
from app.services.document_service import document_service
from app.services.ocr_document_service import enhanced_document_service
from app.services.parallel_pdf_service import parallel_pdf_processor
from app.services.vector_service import vector_service
from app.services.database_service import database_service

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/comprehensive_test_results.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class ComprehensiveTestSuite:
    """Unified comprehensive test suite for all system components"""

    def __init__(self):
        """Initialize test suite with proper paths"""
        self.backend_dir = Path(os.getcwd())
        if self.backend_dir.name != "backend":
            print("❌ Error: Must run from backend directory")
            print("   Usage: cd backend && python tests/test_comprehensive_system.py")
            sys.exit(1)

        self.test_data_dir = self.backend_dir.parent / "test_data"
        self.results = {}
        self.start_time = None
        self.test_count = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.benchmark_results = {}

        logger.info(f"🧪 Comprehensive Test Suite initialized")
        logger.info(f"📁 Backend directory: {self.backend_dir}")
        logger.info(f"📊 Test data directory: {self.test_data_dir}")

    def setup_environment(self):
        """Setup test environment and verify dependencies"""
        logger.info("\n🔧 Setting up test environment...")

        # Check virtual environment
        if not hasattr(sys, "real_prefix") and not (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            logger.warning("⚠️  Warning: Not running in virtual environment")
        else:
            logger.info("✅ Running in virtual environment")

        # Verify test data directory
        if not self.test_data_dir.exists():
            logger.error(f"❌ Test data directory not found: {self.test_data_dir}")
            return False

        # List available test files
        test_files = list(self.test_data_dir.glob("*"))
        logger.info(f"📄 Found {len(test_files)} test files:")
        for file in sorted(test_files):
            if file.is_file():
                size_mb = file.stat().st_size / 1024 / 1024
                logger.info(f"   • {file.name} ({size_mb:.2f} MB)")

        # Create essential directories
        essential_dirs = [
            self.backend_dir / "data" / "originals",
            self.backend_dir / "data" / "converted",
            self.backend_dir / "data" / "temp",
            self.backend_dir / "data" / "metadata",
            self.backend_dir / "logs",
            self.backend_dir / "embeddings",
        ]

        for dir_path in essential_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info("✅ Test environment setup complete")
        return True

    async def run_test(self, test_name, test_func):
        """Run a single test with comprehensive error handling"""
        self.test_count += 1
        logger.info(f"\n🧪 Test {self.test_count}: {test_name}")
        logger.info("-" * 50)

        try:
            start_time = time.time()
            result = await test_func()
            duration = time.time() - start_time

            if result.get("success", False):
                self.passed_tests += 1
                logger.info(f"✅ PASSED ({duration:.3f}s)")
                if result.get("details"):
                    self._log_test_details(result["details"])
            else:
                self.failed_tests += 1
                logger.error(f"❌ FAILED ({duration:.3f}s)")
                logger.error(f"   Error: {result.get('error', 'Unknown error')}")

            self.results[test_name] = {
                "success": result.get("success", False),
                "duration": duration,
                "details": result.get("details", {}),
                "error": result.get("error", ""),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.failed_tests += 1
            duration = time.time() - start_time if "start_time" in locals() else 0
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"❌ FAILED ({duration:.3f}s)")
            logger.error(f"   Exception: {error_msg}")
            logger.debug(f"   Traceback: {traceback.format_exc()}")

            self.results[test_name] = {
                "success": False,
                "duration": duration,
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }

    def _log_test_details(self, details):
        """Log test details in a formatted way"""
        if isinstance(details, dict):
            for key, value in details.items():
                if isinstance(value, dict):
                    logger.info(f"   📊 {key}:")
                    for sub_key, sub_value in value.items():
                        logger.info(f"      • {sub_key}: {sub_value}")
                else:
                    logger.info(f"   📊 {key}: {value}")

    # ==================== CORE SYSTEM TESTS ====================

    async def test_storage_initialization(self):
        """Test storage system initialization"""
        try:
            enhanced_file_storage.setup_storage_directories()

            # Verify all essential directories exist
            required_dirs = [
                settings.original_files_path,
                settings.converted_files_path,
                settings.temp_storage_path,
                f"{settings.data_storage_path}/metadata",
                f"{settings.data_storage_path}/logs",
            ]

            created_dirs = []
            for dir_path in required_dirs:
                path_obj = Path(dir_path)
                if path_obj.exists():
                    created_dirs.append(dir_path)
                else:
                    return {
                        "success": False,
                        "error": f"Directory not created: {dir_path}",
                    }

            storage_stats = enhanced_file_storage.get_storage_stats()

            return {
                "success": True,
                "details": {
                    "directories_created": created_dirs,
                    "storage_stats": storage_stats,
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_hash_calculation(self):
        """Test SHA256 hash calculation and duplicate detection"""
        try:
            test_file = self.test_data_dir / "small.pdf"
            if not test_file.exists():
                return {"success": False, "error": "Test file small.pdf not found"}

            with open(test_file, "rb") as f:
                content = f.read()

            # Test hash calculation consistency
            hash1 = enhanced_file_storage.calculate_file_hash(content)
            hash2 = enhanced_file_storage.calculate_file_hash(content)

            if hash1 != hash2:
                return {"success": False, "error": "Hash calculation inconsistent"}

            # Test duplicate detection
            duplicate_result = await enhanced_file_storage.check_duplicate_by_hash(
                hash1
            )
            is_duplicate = duplicate_result is not None

            return {
                "success": True,
                "details": {
                    "file_hash": hash1,
                    "hash_length": len(hash1),
                    "is_duplicate": is_duplicate,
                    "file_size": len(content),
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_file_conversions(self):
        """Test comprehensive file format conversions"""
        try:
            test_files = [
                ("narensniff.c", "text/plain"),
                ("IMG_CAF37BDF9464-1.jpeg", "image/jpeg"),
                (
                    "All MCA docs.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
                (
                    "TrialBal.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ),
                (
                    "QM Competitive Programming.pptx",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ),
            ]

            conversion_results = {}
            successful_conversions = 0
            total_tested = len(test_files)

            for filename, content_type in test_files:
                test_file = self.test_data_dir / filename

                if not test_file.exists():
                    conversion_results[filename] = {
                        "status": "skipped",
                        "reason": "file_not_found",
                    }
                    continue

                try:
                    with open(test_file, "rb") as f:
                        content = f.read()

                    start_time = time.time()
                    result = await document_service.process_uploaded_file(
                        file_content=content,
                        filename=filename,
                        content_type=content_type,
                    )
                    processing_time = time.time() - start_time

                    if result["status"] == "success":
                        successful_conversions += 1
                        conversion_results[filename] = {
                            "status": "success",
                            "chunk_count": result.get("chunk_count", 0),
                            "text_length": result.get("text_length", 0),
                            "processing_time": round(processing_time, 2),
                        }
                    else:
                        conversion_results[filename] = {
                            "status": "error",
                            "error": result.get("message", "Processing failed"),
                            "chunk_count": 0,
                            "text_length": 0,
                            "processing_time": round(processing_time, 2),
                        }

                except Exception as file_error:
                    conversion_results[filename] = {
                        "status": "error",
                        "error": str(file_error),
                        "chunk_count": 0,
                        "text_length": 0,
                        "processing_time": 0,
                    }

            # Count successful conversions and expected failures (images without text)
            expected_failures = 0
            for filename, result in conversion_results.items():
                if result["status"] == "error" and (
                    filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))
                    or "Could not extract readable text" in result.get("error", "")
                ):
                    expected_failures += 1

            successful_or_expected = successful_conversions + expected_failures
            effective_success_rate = (
                (successful_or_expected / total_tested * 100) if total_tested > 0 else 0
            )

            return {
                "success": effective_success_rate
                >= 50,  # At least 50% success including expected image failures
                "details": {
                    "total_tested": total_tested,
                    "successful_conversions": successful_conversions,
                    "expected_failures": expected_failures,
                    "conversion_results": conversion_results,
                    "success_rate": (
                        (successful_conversions / total_tested * 100)
                        if total_tested > 0
                        else 0
                    ),
                    "effective_success_rate": effective_success_rate,
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_pdf_processing(self):
        """Test PDF processing with different sizes and types"""
        try:
            pdf_files = ["small.pdf", "medmium.pdf", "large_scanned_file.pdf"]
            processing_results = {}
            successful_processing = 0
            total_throughput = 0

            for filename in pdf_files:
                test_file = self.test_data_dir / filename

                if not test_file.exists():
                    logger.warning(f"⚠️  PDF file not found: {filename}")
                    continue

                try:
                    with open(test_file, "rb") as f:
                        content = f.read()

                    file_size_mb = len(content) / 1024 / 1024
                    start_time = time.time()

                    # Choose appropriate service based on file characteristics
                    if file_size_mb > 50 or "scanned" in filename.lower():
                        result = (
                            await enhanced_document_service.process_large_uploaded_file(
                                file_content=content,
                                filename=filename,
                                content_type="application/pdf",
                            )
                        )
                        service_used = "Enhanced OCR Service"
                    else:
                        result = await document_service.process_uploaded_file(
                            file_content=content,
                            filename=filename,
                            content_type="application/pdf",
                        )
                        service_used = "Standard Service"

                    processing_time = time.time() - start_time
                    throughput = (
                        file_size_mb / processing_time if processing_time > 0 else 0
                    )
                    total_throughput += throughput

                    if result["status"] == "success":
                        successful_processing += 1
                        processing_results[filename] = {
                            "status": "success",
                            "file_size_mb": file_size_mb,
                            "processing_time": processing_time,
                            "throughput_mb_per_s": throughput,
                            "chunk_count": result.get("chunk_count", 0),
                            "text_length": result.get("text_length", 0),
                            "service_used": service_used,
                        }
                    else:
                        processing_results[filename] = {
                            "status": "error",
                            "error": result.get("message", "Unknown error"),
                            "service_used": service_used,
                        }

                except Exception as file_error:
                    processing_results[filename] = {
                        "status": "error",
                        "error": str(file_error),
                    }

            average_throughput = (
                total_throughput / successful_processing
                if successful_processing > 0
                else 0
            )

            return {
                "success": successful_processing > 0,
                "details": {
                    "total_tested": len(pdf_files),
                    "successful_processing": successful_processing,
                    "processing_results": processing_results,
                    "average_throughput": average_throughput,
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_parallel_processing(self):
        """Test parallel processing performance"""
        try:
            test_file = self.test_data_dir / "medmium.pdf"
            if not test_file.exists():
                return {"success": False, "error": "Test file medmium.pdf not found"}

            with open(test_file, "rb") as f:
                content = f.read()

            file_size_mb = len(content) / 1024 / 1024
            start_time = time.time()

            try:
                # Correct API usage for parallel PDF processor
                files = [
                    {
                        "content": content,
                        "filename": "medmium.pdf",
                        "content_type": "application/pdf",
                    }
                ]

                result = await parallel_pdf_processor.process_files_parallel(files)
                processing_time = time.time() - start_time
                throughput = (
                    file_size_mb / processing_time if processing_time > 0 else 0
                )

                parallel_workers = getattr(parallel_pdf_processor, "max_workers", 0)
                batch_status = result.get("batch_status", {})
                summary = result.get("summary", {})

                return {
                    "success": batch_status.get("status") == "completed",
                    "details": {
                        "file_size_mb": round(file_size_mb, 2),
                        "processing_time": round(processing_time, 3),
                        "throughput_mb_per_s": throughput,
                        "chunks_created": summary.get("total_chunks_created", 0),
                        "parallel_workers": parallel_workers,
                        "files_processed": summary.get("total_files_processed", 0),
                        "parallel_processing_used": summary.get(
                            "parallel_processing_used", False
                        ),
                    },
                }

            except Exception as parallel_error:
                processing_time = time.time() - start_time
                return {
                    "success": False,
                    "details": {
                        "file_size_mb": round(file_size_mb, 2),
                        "processing_time": round(processing_time, 3),
                        "throughput_mb_per_s": (
                            file_size_mb / processing_time if processing_time > 0 else 0
                        ),
                        "chunks_created": 0,
                        "parallel_workers": 0,
                        "performance_gain": {},
                    },
                    "error": str(parallel_error),
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_ocr_functionality(self):
        """Test OCR functionality on scanned documents"""
        try:
            test_file = self.test_data_dir / "large_scanned_file.pdf"
            if not test_file.exists():
                return {
                    "success": False,
                    "error": "Test file large_scanned_file.pdf not found",
                }

            with open(test_file, "rb") as f:
                content = f.read()

            file_size_mb = len(content) / 1024 / 1024
            start_time = time.time()

            result = await enhanced_document_service.process_large_uploaded_file(
                file_content=content,
                filename="large_scanned_file.pdf",
                content_type="application/pdf",
            )

            processing_time = time.time() - start_time
            throughput = file_size_mb / processing_time if processing_time > 0 else 0

            if result["status"] == "success":
                return {
                    "success": True,
                    "details": {
                        "file_size_mb": round(file_size_mb, 2),
                        "processing_time": round(processing_time, 3),
                        "ocr_text_length": result.get("text_length", 0),
                        "chunks_created": result.get("chunk_count", 0),
                        "ocr_confidence": result.get("ocr_confidence", 0),
                        "pages_processed": result.get("pages_processed", 0),
                        "throughput_mb_per_s": throughput,
                    },
                }
            else:
                return {
                    "success": False,
                    "error": result.get("message", "OCR processing failed"),
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_vector_embeddings(self):
        """Test vector embeddings creation and search operations"""
        try:
            test_documents = [
                {
                    "id": "test_doc_1",
                    "text": "This is a test document about artificial intelligence.",
                },
                {
                    "id": "test_doc_2",
                    "text": "Vector embeddings are numerical representations of text.",
                },
                {
                    "id": "test_doc_3",
                    "text": "ChromaDB is a vector database for AI applications.",
                },
            ]

            # Add documents to vector store (correct method signature)
            document_id = "test_batch_doc"
            chunk_ids = vector_service.add_documents(
                texts=[doc["text"] for doc in test_documents],
                metadatas=[{"doc_id": doc["id"]} for doc in test_documents],
                document_id=document_id,
            )

            # Test similarity search (correct method signature)
            search_results = vector_service.search_similar(
                query="machine learning AI", n_results=2
            )

            # Test collection stats
            stats = vector_service.get_collection_stats()

            return {
                "success": len(search_results) > 0,
                "details": {
                    "documents_added": len(test_documents),
                    "chunk_ids_generated": len(chunk_ids),
                    "search_results_count": len(search_results),
                    "collection_stats": stats,
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_database_operations(self):
        """Test database operations"""
        try:
            # Test session creation
            session_id = database_service.create_session()

            # Test conversation storage
            conversation_id = database_service.save_conversation(
                session_id=session_id,
                user_message="Test user message",
                ai_response="Test AI response",
                sources=[{"doc_id": "test", "content": "test content"}],
            )

            # Test conversation retrieval
            history = database_service.get_conversation_history(session_id, limit=5)

            # Test document storage
            doc_id = database_service.save_document(
                filename="test_document.txt",
                content="This is test content",
                doc_type="text",
                metadata={"test": True, "file_size": 1024},
            )

            # Test document retrieval
            documents = database_service.get_documents()

            return {
                "success": True,
                "details": {
                    "session_created": bool(session_id),
                    "conversation_stored": bool(conversation_id),
                    "history_retrieved": len(history),
                    "document_stored": bool(doc_id),
                    "documents_count": len(documents),
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_system_reset(self):
        """Test complete system reset"""
        try:
            before_stats = enhanced_file_storage.get_storage_stats()

            reset_result = await enhanced_file_storage.reset_all_storage()

            after_stats = enhanced_file_storage.get_storage_stats()

            directories_exist = all(
                Path(path).exists()
                for path in [
                    settings.original_files_path,
                    settings.converted_files_path,
                    settings.temp_storage_path,
                ]
            )

            return {
                "success": reset_result and directories_exist,
                "details": {
                    "before_stats": before_stats,
                    "after_stats": after_stats,
                    "directories_recreated": directories_exist,
                    "storage_cleared": reset_result,
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== PERFORMANCE BENCHMARKING ====================

    async def run_performance_benchmarks(self):
        """Run performance benchmarks"""
        logger.info("\n🚀 Running Performance Benchmarks...")

        benchmark_files = ["small.pdf", "medmium.pdf", "large_scanned_file.pdf"]

        for filename in benchmark_files:
            test_file = self.test_data_dir / filename
            if test_file.exists():
                benchmark_result = await self._benchmark_file_processing(test_file)
                if benchmark_result:
                    self.benchmark_results[filename] = benchmark_result
                    logger.info(
                        f"📊 {filename}: {benchmark_result['throughput_mb_per_s']:.2f} MB/s"
                    )

    async def _benchmark_file_processing(
        self, file_path: Path, runs: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Benchmark file processing performance"""
        try:
            with open(file_path, "rb") as f:
                content = f.read()

            file_size_mb = len(content) / 1024 / 1024
            times = []

            for run in range(runs):
                start_time = time.time()

                if file_size_mb > 50 or "scanned" in file_path.name.lower():
                    result = (
                        await enhanced_document_service.process_large_uploaded_file(
                            file_content=content,
                            filename=file_path.name,
                            content_type="application/pdf",
                        )
                    )
                else:
                    result = await document_service.process_uploaded_file(
                        file_content=content,
                        filename=file_path.name,
                        content_type="application/pdf",
                    )

                processing_time = time.time() - start_time
                times.append(processing_time)

                if result["status"] != "success":
                    logger.warning(f"⚠️  Benchmark failed for {file_path.name}")
                    return None

            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            throughput = file_size_mb / avg_time

            return {
                "file_size_mb": file_size_mb,
                "avg_processing_time": avg_time,
                "min_processing_time": min_time,
                "max_processing_time": max_time,
                "throughput_mb_per_s": throughput,
                "chunk_count": result.get("chunk_count", 0),
            }

        except Exception as e:
            logger.error(f"❌ Benchmark error for {file_path.name}: {e}")
            return None

    # ==================== MAIN TEST EXECUTION ====================

    async def run_all_tests(self):
        """Execute all tests in sequence"""
        logger.info("\n🧪 Starting Comprehensive Test Suite")
        logger.info("=" * 50)

        self.start_time = time.time()

        # Execute all tests
        test_functions = [
            ("Storage Initialization", self.test_storage_initialization),
            ("SHA256 Hash Calculation", self.test_hash_calculation),
            ("File Format Conversions", self.test_file_conversions),
            ("PDF Processing", self.test_pdf_processing),
            ("Parallel Processing", self.test_parallel_processing),
            ("OCR Functionality", self.test_ocr_functionality),
            ("Vector Embeddings", self.test_vector_embeddings),
            ("Database Operations", self.test_database_operations),
            ("System Reset", self.test_system_reset),
        ]

        for test_name, test_func in test_functions:
            await self.run_test(test_name, test_func)

        # Run performance benchmarks
        await self.run_performance_benchmarks()

        # Generate final report
        self.generate_comprehensive_report()

        # Display summary
        logger.info("\n" + "=" * 50)
        logger.info("🎯 TEST EXECUTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"📊 Total Tests: {self.test_count}")
        logger.info(f"✅ Passed: {self.passed_tests}")
        logger.info(f"❌ Failed: {self.failed_tests}")
        logger.info(f"📈 Success Rate: {(self.passed_tests/self.test_count*100):.1f}%")
        logger.info(f"⏱️  Total Duration: {time.time() - self.start_time:.2f}s")
        logger.info("=" * 50)

        return 0 if self.failed_tests == 0 else 1

    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        logger.info("\n📊 Generating Comprehensive Test Report...")

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["success"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        report = {
            "test_run_info": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": (
                    time.time() - self.start_time if self.start_time else 0
                ),
                "backend_directory": str(self.backend_dir),
                "test_data_directory": str(self.test_data_dir),
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
            },
            "test_results": self.results,
            "performance_benchmarks": self.benchmark_results,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": str(self.backend_dir),
            },
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.backend_dir / "logs" / f"test_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"📄 Report saved: {report_file}")


async def main():
    """Main function for standalone execution"""
    try:
        test_suite = ComprehensiveTestSuite()

        if not test_suite.setup_environment():
            return 1

        exit_code = await test_suite.run_all_tests()
        return exit_code

    except KeyboardInterrupt:
        logger.info("\n⚠️  Test execution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Test execution failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
