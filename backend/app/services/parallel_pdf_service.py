"""
Parallel PDF Processing Service
Splits large PDFs into smaller chunks and processes them in parallel with real-time progress tracking
"""

import asyncio
import aiofiles
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
import io
import sys
import tempfile
import math
import multiprocessing
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import time
import os
import pickle

# Document processing imports
import PyPDF2

try:
    import fitz  # PyMuPDF for better PDF handling

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdf2image import convert_from_path, convert_from_bytes
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Internal imports
from app.core.config import settings
from app.services.vector_service import vector_service
from app.services.database_service import DatabaseService

logger = logging.getLogger(__name__)


# Standalone functions for multiprocessing (must be picklable)
def extract_text_from_pdf_chunk(chunk_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract text from a PDF chunk - standalone function for multiprocessing

    Args:
        chunk_data: Dictionary containing chunk information and file content

    Returns:
        Dictionary with extracted text and processing info
    """
    import PyPDF2
    import io
    from datetime import datetime

    try:
        # Try PyMuPDF first if available
        if PYMUPDF_AVAILABLE:
            import fitz

            doc = fitz.open(stream=chunk_data["file_content"], filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()

            if len(text.strip()) > 50:  # Sufficient text found
                return {
                    "success": True,
                    "text": text,
                    "chunk_id": chunk_data["chunk_id"],
                    "pages": f"{chunk_data['start_page']}-{chunk_data['end_page']}",
                    "method": "pymupdf",
                }

        # Fallback to PyPDF2
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(chunk_data["file_content"]))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

        if len(text.strip()) > 50:  # Sufficient text found
            return {
                "success": True,
                "text": text,
                "chunk_id": chunk_data["chunk_id"],
                "pages": f"{chunk_data['start_page']}-{chunk_data['end_page']}",
                "method": "pypdf2",
            }

        # If minimal text, try OCR
        if OCR_AVAILABLE and len(text.strip()) < 50:
            return perform_ocr_on_pdf_chunk(chunk_data)

        return {
            "success": True,
            "text": text,
            "chunk_id": chunk_data["chunk_id"],
            "pages": f"{chunk_data['start_page']}-{chunk_data['end_page']}",
            "method": "minimal_text",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "chunk_id": chunk_data["chunk_id"],
            "pages": f"{chunk_data['start_page']}-{chunk_data['end_page']}",
        }


def perform_ocr_on_pdf_chunk(chunk_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform OCR on a PDF chunk - standalone function for multiprocessing
    """
    try:
        if not OCR_AVAILABLE:
            return {
                "success": False,
                "error": "OCR not available",
                "chunk_id": chunk_data["chunk_id"],
            }

        from pdf2image import convert_from_bytes
        import pytesseract

        # Convert PDF chunk to images
        images = convert_from_bytes(chunk_data["file_content"], dpi=300)

        ocr_text = ""
        for i, image in enumerate(images):
            # Perform OCR on each page
            page_text = pytesseract.image_to_string(image, lang="eng")
            ocr_text += f"Page {chunk_data['start_page'] + i + 1}:\n{page_text}\n\n"

        return {
            "success": True,
            "text": ocr_text,
            "chunk_id": chunk_data["chunk_id"],
            "pages": f"{chunk_data['start_page']}-{chunk_data['end_page']}",
            "method": "ocr",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "chunk_id": chunk_data["chunk_id"],
            "pages": f"{chunk_data['start_page']}-{chunk_data['end_page']}",
        }


def create_text_chunks_from_content(
    text_content: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Create text chunks from content - standalone function for multiprocessing
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        if text_content.strip():
            documents = text_splitter.split_text(text_content)
            chunks = []
            for i, doc in enumerate(documents):
                chunks.append(
                    {
                        "content": doc,
                        "metadata": {"chunk_index": i, "total_chunks": len(documents)},
                    }
                )
            return chunks
        else:
            return []

    except Exception as e:
        logger.error(f"Failed to create text chunks: {e}")
        return []


@dataclass
class ProcessingChunk:
    """Represents a chunk of PDF to be processed"""

    chunk_id: str
    start_page: int
    end_page: int
    total_pages: int
    file_content: bytes
    estimated_time: float
    status: str = "pending"
    progress: float = 0.0
    text_extracted: str = ""
    chunks_created: int = 0
    error_message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class FileProcessingStatus:
    """Tracks processing status for an entire file"""

    file_id: str
    filename: str
    file_size: int
    total_pages: int
    chunks: List[ProcessingChunk]
    overall_progress: float = 0.0
    estimated_total_time: float = 0.0
    elapsed_time: float = 0.0
    remaining_time: float = 0.0
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    parallel_workers: int = 4


class ParallelPDFProcessor:
    """Handles parallel processing of large PDFs with real-time progress tracking"""

    def __init__(self):
        self.db_service = DatabaseService()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        self.processing_status: Dict[str, FileProcessingStatus] = {}
        # Use all available CPU cores, but cap at 8 for memory efficiency
        self.max_workers = min(multiprocessing.cpu_count(), 8)

        # Adaptive chunking parameters (replacing fixed 20 pages)
        self.adaptive_chunking = {
            "min_pages_per_chunk": 5,  # Minimum pages per chunk
            "max_pages_per_chunk": 50,  # Maximum pages per chunk
            "target_chunk_time": 15,  # Target processing time per chunk (seconds)
            "max_memory_per_chunk": 200,  # Maximum MB per chunk
            "optimal_chunks_per_core": 1.5,  # Chunks per CPU core for load balancing
        }

        self.time_threshold = 30  # 30 seconds threshold for parallel processing

        # Create process pool for CPU-intensive tasks
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)

        logger.info(
            f"Initialized parallel processor with {self.max_workers} CPU cores and adaptive chunking"
        )

    async def process_files_parallel(
        self, files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process multiple files in parallel with real-time progress tracking

        Args:
            files: List of file dictionaries with 'content', 'filename', 'content_type'

        Returns:
            Processing status with real-time updates
        """
        batch_id = str(uuid.uuid4())
        start_time = datetime.now()

        # Initialize batch status
        batch_status = {
            "batch_id": batch_id,
            "total_files": len(files),
            "completed_files": 0,
            "failed_files": 0,
            "overall_progress": 0.0,
            "estimated_total_time": 0.0,
            "elapsed_time": 0.0,
            "remaining_time": 0.0,
            "status": "initializing",
            "files": {},
            "start_time": start_time.isoformat(),
            "parallel_processing": True,
        }

        try:
            # Analyze all files and estimate processing time
            logger.info(f"Analyzing {len(files)} files for parallel processing")

            file_analyses = []
            total_estimated_time = 0.0

            for file_data in files:
                analysis = await self._analyze_file_for_processing(file_data)
                file_analyses.append(analysis)
                total_estimated_time += analysis["estimated_time"]

                # Add to batch status
                batch_status["files"][analysis["file_id"]] = {
                    "filename": analysis["filename"],
                    "file_size": analysis["file_size"],
                    "estimated_time": analysis["estimated_time"],
                    "requires_parallel": analysis["requires_parallel"],
                    "total_pages": analysis.get("total_pages", 0),
                    "parallel_chunks": analysis.get("parallel_chunks", 1),
                    "status": "pending",
                    "progress": 0.0,
                }

            batch_status["estimated_total_time"] = total_estimated_time
            batch_status["status"] = "processing"

            # Process files in parallel
            logger.info(f"Starting parallel processing of {len(files)} files")

            # Create processing tasks
            processing_tasks = []
            for analysis in file_analyses:
                if analysis["requires_parallel"]:
                    task = self._process_large_file_parallel(analysis, batch_status)
                else:
                    task = self._process_regular_file(analysis, batch_status)
                processing_tasks.append(task)

            # Execute all tasks in parallel
            results = await asyncio.gather(*processing_tasks, return_exceptions=True)

            # Update final status
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]

            batch_status["completed_files"] = len(successful_results)
            batch_status["failed_files"] = len(failed_results)
            batch_status["overall_progress"] = 100.0
            batch_status["status"] = (
                "completed" if not failed_results else "completed_with_errors"
            )
            batch_status["end_time"] = datetime.now().isoformat()
            batch_status["elapsed_time"] = (datetime.now() - start_time).total_seconds()

            # Compile final results
            total_chunks = sum(
                len(r.get("chunks", []))
                for r in successful_results
                if isinstance(r, dict)
            )
            total_characters = sum(
                r.get("text_length", 0)
                for r in successful_results
                if isinstance(r, dict)
            )

            return {
                "batch_status": batch_status,
                "results": successful_results,
                "errors": [str(e) for e in failed_results],
                "summary": {
                    "total_files_processed": len(successful_results),
                    "total_chunks_created": total_chunks,
                    "total_characters_extracted": total_characters,
                    "total_processing_time": batch_status["elapsed_time"],
                    "average_time_per_file": (
                        batch_status["elapsed_time"] / len(files) if files else 0
                    ),
                    "parallel_processing_used": any(
                        a["requires_parallel"] for a in file_analyses
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            batch_status["status"] = "failed"
            batch_status["error"] = str(e)
            return {"batch_status": batch_status, "error": str(e)}

    async def _analyze_file_for_processing(
        self, file_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a file to determine if parallel processing is needed"""
        file_id = str(uuid.uuid4())
        filename = file_data["filename"]
        file_content = file_data["content"]
        file_size = len(file_content)

        analysis = {
            "file_id": file_id,
            "filename": filename,
            "file_size": file_size,
            "file_content": file_content,
            "content_type": file_data.get("content_type", ""),
            "requires_parallel": False,
            "estimated_time": 0.0,
            "total_pages": 0,
            "parallel_chunks": 1,
            "document_characteristics": {},
        }

        # Estimate processing time based on file size and type
        if filename.lower().endswith(".pdf"):
            # Get page count and analyze document characteristics
            try:
                if PYMUPDF_AVAILABLE:
                    doc = fitz.open(stream=file_content, filetype="pdf")
                    total_pages = len(doc)

                    # Analyze document characteristics
                    doc_analysis = await self._analyze_pdf_characteristics(
                        doc, file_size
                    )
                    analysis["document_characteristics"] = doc_analysis
                    doc.close()
                else:
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                    total_pages = len(pdf_reader.pages)

                    # Basic analysis without PyMuPDF
                    doc_analysis = {
                        "avg_page_size_mb": file_size / (1024 * 1024) / total_pages,
                        "is_likely_scanned": file_size > 50 * 1024 * 1024,
                        "complexity_score": 1.0,  # Default
                        "has_images": True,  # Assume true for safety
                    }
                    analysis["document_characteristics"] = doc_analysis

                analysis["total_pages"] = total_pages

                # Calculate optimal chunking strategy
                optimal_strategy = self._calculate_optimal_chunking(
                    total_pages, doc_analysis, file_size
                )

                estimated_time = optimal_strategy["estimated_total_time"]
                analysis["estimated_time"] = estimated_time
                analysis["chunking_strategy"] = optimal_strategy

                # Determine if parallel processing is needed
                if estimated_time > self.time_threshold or total_pages > 50:
                    analysis["requires_parallel"] = True
                    analysis["parallel_chunks"] = optimal_strategy["num_chunks"]
                else:
                    analysis["requires_parallel"] = False
                    analysis["parallel_chunks"] = 1

            except Exception as e:
                logger.warning(f"Could not analyze PDF {filename}: {e}")
                analysis["estimated_time"] = 60.0  # Conservative estimate
                analysis["requires_parallel"] = file_size > 10 * 1024 * 1024  # > 10MB
        else:
            # Non-PDF files - estimate based on size
            analysis["estimated_time"] = max(
                5.0, file_size / (1024 * 1024) * 2
            )  # 2 seconds per MB
            analysis["requires_parallel"] = file_size > 50 * 1024 * 1024  # > 50MB

        return analysis

    async def _analyze_pdf_characteristics(self, doc, file_size: int) -> Dict[str, Any]:
        """Analyze PDF characteristics to optimize chunking strategy"""
        total_pages = len(doc)

        # Sample first few pages to understand document characteristics
        sample_pages = min(5, total_pages)
        total_text_length = 0
        total_images = 0
        total_objects = 0

        for page_num in range(sample_pages):
            page = doc[page_num]

            # Count text
            text = page.get_text()
            total_text_length += len(text)

            # Count images
            image_list = page.get_images()
            total_images += len(image_list)

            # Count objects (rough complexity measure)
            total_objects += len(page.get_contents())

        # Calculate averages and characteristics
        avg_text_per_page = total_text_length / sample_pages if sample_pages > 0 else 0
        avg_images_per_page = total_images / sample_pages if sample_pages > 0 else 0
        avg_objects_per_page = total_objects / sample_pages if sample_pages > 0 else 0
        avg_page_size_mb = file_size / (1024 * 1024) / total_pages

        # Determine document type and complexity
        is_text_heavy = avg_text_per_page > 1000
        is_image_heavy = avg_images_per_page > 2 or avg_page_size_mb > 5
        is_likely_scanned = avg_text_per_page < 100 and avg_page_size_mb > 2

        # Calculate complexity score (1.0 = simple, 3.0 = very complex)
        complexity_score = 1.0
        if is_image_heavy:
            complexity_score += 0.5
        if is_likely_scanned:
            complexity_score += 1.0
        if avg_objects_per_page > 50:
            complexity_score += 0.5

        return {
            "avg_text_per_page": avg_text_per_page,
            "avg_images_per_page": avg_images_per_page,
            "avg_page_size_mb": avg_page_size_mb,
            "is_text_heavy": is_text_heavy,
            "is_image_heavy": is_image_heavy,
            "is_likely_scanned": is_likely_scanned,
            "complexity_score": min(complexity_score, 3.0),
            "has_images": avg_images_per_page > 0,
        }

    def _calculate_optimal_chunking(
        self, total_pages: int, doc_characteristics: Dict[str, Any], file_size: int
    ) -> Dict[str, Any]:
        """Calculate optimal chunking strategy based on document characteristics"""

        # Base processing time estimates per page
        base_time_per_page = 0.3  # Fast text extraction
        ocr_time_per_page = 2.5  # OCR processing
        complex_time_per_page = 1.0  # Complex documents

        # Determine processing time per page based on characteristics
        complexity_score = doc_characteristics.get("complexity_score", 1.0)
        is_likely_scanned = doc_characteristics.get("is_likely_scanned", False)
        avg_page_size_mb = doc_characteristics.get("avg_page_size_mb", 1.0)

        if is_likely_scanned:
            time_per_page = ocr_time_per_page * complexity_score
        elif complexity_score > 2.0:
            time_per_page = complex_time_per_page * complexity_score
        else:
            time_per_page = base_time_per_page * complexity_score

        # Calculate memory usage per page (rough estimate)
        memory_per_page_mb = max(1.0, avg_page_size_mb * 2)  # Processing overhead
        if is_likely_scanned:
            memory_per_page_mb *= 3  # OCR memory overhead

        # Calculate optimal pages per chunk based on constraints
        max_pages_by_time = max(
            self.adaptive_chunking["min_pages_per_chunk"],
            int(self.adaptive_chunking["target_chunk_time"] / time_per_page),
        )

        max_pages_by_memory = max(
            self.adaptive_chunking["min_pages_per_chunk"],
            int(self.adaptive_chunking["max_memory_per_chunk"] / memory_per_page_mb),
        )

        # Use the most restrictive constraint
        optimal_pages_per_chunk = min(
            max_pages_by_time,
            max_pages_by_memory,
            self.adaptive_chunking["max_pages_per_chunk"],
        )

        # Calculate number of chunks
        ideal_num_chunks = math.ceil(total_pages / optimal_pages_per_chunk)

        # Optimize for CPU utilization
        optimal_chunks_total = int(
            self.max_workers * self.adaptive_chunking["optimal_chunks_per_core"]
        )

        # Balance between ideal chunking and CPU utilization
        if ideal_num_chunks < 2:
            num_chunks = 1  # Too small for parallel processing
        elif ideal_num_chunks <= optimal_chunks_total:
            num_chunks = ideal_num_chunks
        else:
            # Too many chunks, balance between CPU cores and chunk size
            num_chunks = min(ideal_num_chunks, optimal_chunks_total)

        # Recalculate actual pages per chunk
        actual_pages_per_chunk = (
            math.ceil(total_pages / num_chunks) if num_chunks > 0 else total_pages
        )

        # Calculate estimates
        estimated_time_per_chunk = actual_pages_per_chunk * time_per_page
        estimated_total_time = (
            estimated_time_per_chunk
            if num_chunks == 1
            else estimated_time_per_chunk * 1.2
        )  # Parallel overhead
        estimated_memory_per_chunk = actual_pages_per_chunk * memory_per_page_mb

        strategy = {
            "num_chunks": max(1, num_chunks),
            "pages_per_chunk": actual_pages_per_chunk,
            "estimated_time_per_chunk": estimated_time_per_chunk,
            "estimated_total_time": estimated_total_time,
            "estimated_memory_per_chunk_mb": estimated_memory_per_chunk,
            "processing_method": "ocr" if is_likely_scanned else "text_extraction",
            "optimization_factors": {
                "time_constraint": max_pages_by_time,
                "memory_constraint": max_pages_by_memory,
                "cpu_utilization": optimal_chunks_total,
                "chosen_constraint": (
                    "time" if max_pages_by_time <= max_pages_by_memory else "memory"
                ),
            },
        }

        logger.info(
            f"Optimal chunking strategy: {num_chunks} chunks, "
            f"{actual_pages_per_chunk} pages/chunk, "
            f"~{estimated_time_per_chunk:.1f}s/chunk, "
            f"~{estimated_memory_per_chunk:.1f}MB/chunk"
        )

        return strategy

    async def _process_large_file_parallel(
        self, analysis: Dict[str, Any], batch_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a large file using parallel processing"""
        file_id = analysis["file_id"]
        filename = analysis["filename"]
        file_content = analysis["file_content"]
        total_pages = analysis["total_pages"]

        logger.info(f"Starting parallel processing of {filename} ({total_pages} pages)")

        try:
            # Split PDF into chunks using adaptive strategy
            pdf_chunks = await self._split_pdf_into_chunks_adaptive(
                file_content, filename, analysis["chunking_strategy"]
            )

            # Create processing status
            file_status = FileProcessingStatus(
                file_id=file_id,
                filename=filename,
                file_size=len(file_content),
                total_pages=total_pages,
                chunks=pdf_chunks,
                parallel_workers=len(pdf_chunks),
                start_time=datetime.now(),
            )

            self.processing_status[file_id] = file_status

            # Update batch status
            batch_status["files"][file_id]["status"] = "processing"
            batch_status["files"][file_id]["parallel_chunks"] = len(pdf_chunks)

            # Process chunks using multiprocessing for true parallelism
            logger.info(
                f"Processing {len(pdf_chunks)} chunks using {self.max_workers} CPU cores"
            )

            # Prepare chunk data for multiprocessing
            chunk_data_list = []
            for chunk in pdf_chunks:
                chunk_data = {
                    "chunk_id": chunk.chunk_id,
                    "start_page": chunk.start_page,
                    "end_page": chunk.end_page,
                    "file_content": chunk.file_content,
                }
                chunk_data_list.append(chunk_data)

            # Submit tasks to process pool
            loop = asyncio.get_event_loop()
            future_to_chunk = {}

            for i, chunk_data in enumerate(chunk_data_list):
                future = self.process_pool.submit(
                    extract_text_from_pdf_chunk, chunk_data
                )
                future_to_chunk[future] = (i, pdf_chunks[i])

            # Collect results as they complete
            chunk_results = [None] * len(pdf_chunks)
            completed_count = 0

            for future in as_completed(future_to_chunk):
                chunk_index, chunk = future_to_chunk[future]

                try:
                    result = future.result()
                    chunk_results[chunk_index] = result

                    # Update chunk status
                    if result["success"]:
                        chunk.status = "completed"
                        chunk.progress = 100.0
                        chunk.text_extracted = result["text"]

                        # Create text chunks using multiprocessing
                        text_chunks = await loop.run_in_executor(
                            self.process_pool,
                            create_text_chunks_from_content,
                            result["text"],
                            settings.chunk_size,
                            settings.chunk_overlap,
                        )
                        chunk.chunks_created = len(text_chunks)
                        result["text_chunks"] = text_chunks
                    else:
                        chunk.status = "failed"
                        chunk.error_message = result.get("error", "Unknown error")

                    chunk.end_time = datetime.now()
                    completed_count += 1

                    # Update file-level progress
                    file_status.overall_progress = (
                        completed_count / len(pdf_chunks)
                    ) * 100
                    batch_status["files"][file_id][
                        "progress"
                    ] = file_status.overall_progress

                    logger.info(
                        f"Completed chunk {chunk_index + 1}/{len(pdf_chunks)} - {result.get('method', 'unknown')} method"
                    )

                except Exception as e:
                    logger.error(f"Chunk {chunk_index} processing failed: {e}")
                    chunk.status = "failed"
                    chunk.error_message = str(e)
                    chunk.end_time = datetime.now()
                    chunk_results[chunk_index] = {"success": False, "error": str(e)}

                    # Combine results
            all_text = ""
            all_chunks = []
            total_chunk_count = 0
            successful_chunks = 0

            for i, result in enumerate(chunk_results):
                if result and result.get("success"):
                    all_text += result["text"] + "\n\n"

                    # Convert text chunks to LangChain Documents
                    if "text_chunks" in result:
                        for chunk_data in result["text_chunks"]:
                            langchain_doc = Document(
                                page_content=chunk_data["content"],
                                metadata={
                                    "source": f"{filename}_chunk_{pdf_chunks[i].chunk_id}",
                                    "chunk_index": chunk_data["metadata"][
                                        "chunk_index"
                                    ],
                                    "pdf_pages": f"{pdf_chunks[i].start_page}-{pdf_chunks[i].end_page}",
                                    "total_chunks": chunk_data["metadata"][
                                        "total_chunks"
                                    ],
                                    "processing_method": result.get(
                                        "method", "unknown"
                                    ),
                                },
                            )
                            all_chunks.append(langchain_doc)

                    total_chunk_count += len(result.get("text_chunks", []))
                    successful_chunks += 1
                else:
                    logger.error(
                        f"Chunk {i} failed: {result.get('error', 'Unknown error') if result else 'No result'}"
                    )

            logger.info(
                f"Successfully processed {successful_chunks}/{len(pdf_chunks)} chunks, created {total_chunk_count} text chunks"
            )

            # Store combined results
            document_id = str(uuid.uuid4())

            # Generate embeddings for all chunks
            if all_chunks:
                await self._store_chunks_and_embeddings(
                    all_chunks, document_id, filename
                )

            # Store document metadata
            await self._store_document_metadata(
                filename=filename,
                text_content=all_text,
                content_type=analysis["content_type"],
                file_size=len(file_content),
                chunk_count=total_chunk_count,
                document_id=document_id,
                processing_method="parallel",
            )

            # Save original file
            await self._save_file_async(file_content, document_id, filename)

            # Update final status
            file_status.status = "completed"
            file_status.end_time = datetime.now()
            file_status.overall_progress = 100.0

            batch_status["files"][file_id]["status"] = "completed"
            batch_status["files"][file_id]["progress"] = 100.0

            processing_time = (
                file_status.end_time - file_status.start_time
            ).total_seconds()

            return {
                "document_id": document_id,
                "filename": filename,
                "status": "success",
                "text_length": len(all_text),
                "chunk_count": total_chunk_count,
                "chunks": all_chunks,
                "processing_time_seconds": processing_time,
                "processing_method": "parallel",
                "parallel_chunks_processed": len(pdf_chunks),
                "pages_processed": total_pages,
            }

        except Exception as e:
            logger.error(f"Parallel processing failed for {filename}: {e}")
            batch_status["files"][file_id]["status"] = "failed"
            batch_status["files"][file_id]["error"] = str(e)
            raise e

    async def _split_pdf_into_chunks_adaptive(
        self, file_content: bytes, filename: str, chunking_strategy: Dict[str, Any]
    ) -> List[ProcessingChunk]:
        """Split a PDF into optimal chunks using adaptive strategy"""
        num_chunks = chunking_strategy["num_chunks"]
        pages_per_chunk = chunking_strategy["pages_per_chunk"]

        logger.info(
            f"Splitting {filename} into {num_chunks} adaptive chunks ({pages_per_chunk} pages/chunk)"
        )

        try:
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(stream=file_content, filetype="pdf")
                total_pages = len(doc)
                doc.close()
            else:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                total_pages = len(pdf_reader.pages)

            chunks = []

            for i in range(num_chunks):
                start_page = i * pages_per_chunk
                end_page = min(start_page + pages_per_chunk - 1, total_pages - 1)

                if start_page <= end_page:
                    # Extract pages for this chunk
                    chunk_content = await self._extract_pdf_pages(
                        file_content, start_page, end_page
                    )

                    chunk = ProcessingChunk(
                        chunk_id=f"{filename}_adaptive_chunk_{i+1}",
                        start_page=start_page,
                        end_page=end_page,
                        total_pages=end_page - start_page + 1,
                        file_content=chunk_content,
                        estimated_time=chunking_strategy["estimated_time_per_chunk"],
                    )
                    chunks.append(chunk)

            logger.info(f"Created {len(chunks)} adaptive chunks from {filename}")
            return chunks

        except Exception as e:
            logger.error(f"Failed to split PDF {filename}: {e}")
            raise e

    async def _extract_pdf_pages(
        self, file_content: bytes, start_page: int, end_page: int
    ) -> bytes:
        """Extract specific pages from a PDF"""
        try:
            if PYMUPDF_AVAILABLE:
                # Use PyMuPDF for better performance
                source_doc = fitz.open(stream=file_content, filetype="pdf")
                new_doc = fitz.open()

                for page_num in range(start_page, end_page + 1):
                    if page_num < len(source_doc):
                        new_doc.insert_pdf(
                            source_doc, from_page=page_num, to_page=page_num
                        )

                chunk_content = new_doc.write()
                new_doc.close()
                source_doc.close()

                return chunk_content
            else:
                # Fallback to PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                pdf_writer = PyPDF2.PdfWriter()

                for page_num in range(start_page, end_page + 1):
                    if page_num < len(pdf_reader.pages):
                        pdf_writer.add_page(pdf_reader.pages[page_num])

                output_buffer = io.BytesIO()
                pdf_writer.write(output_buffer)
                return output_buffer.getvalue()

        except Exception as e:
            logger.error(f"Failed to extract pages {start_page}-{end_page}: {e}")
            raise e

    async def _process_pdf_chunk(
        self,
        chunk: ProcessingChunk,
        file_status: FileProcessingStatus,
        batch_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a single PDF chunk"""
        chunk.start_time = datetime.now()
        chunk.status = "processing"

        try:
            logger.info(
                f"Processing chunk {chunk.chunk_id} (pages {chunk.start_page}-{chunk.end_page})"
            )

            # Extract text from chunk
            chunk.progress = 10.0
            text_content = await self._extract_text_from_chunk(chunk)
            chunk.text_extracted = text_content
            chunk.progress = 60.0

            # Create text chunks
            if text_content.strip():
                documents = self.text_splitter.split_text(text_content)
                langchain_chunks = [
                    Document(
                        page_content=doc,
                        metadata={
                            "source": f"{file_status.filename}_chunk_{chunk.chunk_id}",
                            "chunk_index": i,
                            "pdf_pages": f"{chunk.start_page}-{chunk.end_page}",
                            "total_chunks": len(documents),
                        },
                    )
                    for i, doc in enumerate(documents)
                ]
                chunk.chunks_created = len(langchain_chunks)
            else:
                langchain_chunks = []
                chunk.chunks_created = 0

            chunk.progress = 100.0
            chunk.status = "completed"
            chunk.end_time = datetime.now()

            # Update file-level progress
            completed_chunks = sum(
                1 for c in file_status.chunks if c.status == "completed"
            )
            file_status.overall_progress = (
                completed_chunks / len(file_status.chunks)
            ) * 100

            # Update batch-level progress
            batch_status["files"][file_status.file_id][
                "progress"
            ] = file_status.overall_progress

            return {
                "text": text_content,
                "chunks": langchain_chunks,
                "pages_processed": chunk.total_pages,
            }

        except Exception as e:
            chunk.status = "failed"
            chunk.error_message = str(e)
            chunk.end_time = datetime.now()
            logger.error(f"Failed to process chunk {chunk.chunk_id}: {e}")
            raise e

    async def _extract_text_from_chunk(self, chunk: ProcessingChunk) -> str:
        """Extract text from a PDF chunk with OCR fallback"""
        try:
            # Try PyMuPDF first (faster)
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(stream=chunk.file_content, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
                doc.close()

                if len(text.strip()) > 50:  # Sufficient text found
                    return text

            # Fallback to PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(chunk.file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            if len(text.strip()) > 50:  # Sufficient text found
                return text

            # If minimal text, try OCR
            if OCR_AVAILABLE and len(text.strip()) < 50:
                logger.info(f"Performing OCR on chunk {chunk.chunk_id}")
                return await self._perform_ocr_on_chunk(chunk)

            return text

        except Exception as e:
            logger.error(f"Text extraction failed for chunk {chunk.chunk_id}: {e}")
            return ""

    async def _perform_ocr_on_chunk(self, chunk: ProcessingChunk) -> str:
        """Perform OCR on a PDF chunk"""
        try:
            # Convert PDF chunk to images
            images = convert_from_bytes(chunk.file_content, dpi=300)

            ocr_text = ""
            for i, image in enumerate(images):
                # Perform OCR on each page
                page_text = pytesseract.image_to_string(image, lang="eng")
                ocr_text += f"Page {chunk.start_page + i + 1}:\n{page_text}\n\n"

            return ocr_text

        except Exception as e:
            logger.error(f"OCR failed for chunk {chunk.chunk_id}: {e}")
            return ""

    async def _process_regular_file(
        self, analysis: Dict[str, Any], batch_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a regular file that doesn't need parallel processing"""
        # Import the regular document service for non-parallel processing
        from app.services.document_service import document_service

        file_id = analysis["file_id"]
        batch_status["files"][file_id]["status"] = "processing"

        try:
            result = await document_service.process_uploaded_file(
                file_content=analysis["file_content"],
                filename=analysis["filename"],
                content_type=analysis["content_type"],
            )

            batch_status["files"][file_id]["status"] = "completed"
            batch_status["files"][file_id]["progress"] = 100.0

            return result

        except Exception as e:
            batch_status["files"][file_id]["status"] = "failed"
            batch_status["files"][file_id]["error"] = str(e)
            raise e

    async def _store_chunks_and_embeddings(
        self, chunks: List[Document], document_id: str, filename: str
    ):
        """Store chunks and generate embeddings"""
        try:
            # Generate embeddings in batches
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]

                # Generate embeddings and store
                await vector_service.add_documents_async(texts, metadatas, document_id)

        except Exception as e:
            logger.error(f"Failed to store chunks for {filename}: {e}")
            raise e

    async def _store_document_metadata(
        self,
        filename: str,
        text_content: str,
        content_type: str,
        file_size: int,
        chunk_count: int,
        document_id: str,
        processing_method: str,
    ):
        """Store document metadata in database"""
        try:
            metadata = {
                "filename": filename,
                "content_type": content_type,
                "file_size": file_size,
                "text_length": len(text_content),
                "chunk_count": chunk_count,
                "processing_method": processing_method,
                "upload_date": datetime.now().isoformat(),
                "document_id": document_id,
            }

            await self.db_service.store_document_async(document_id, metadata)

        except Exception as e:
            logger.error(f"Failed to store metadata for {filename}: {e}")
            raise e

    async def _save_file_async(
        self, file_content: bytes, document_id: str, filename: str
    ):
        """Save the original file asynchronously"""
        try:
            file_extension = Path(filename).suffix
            save_filename = f"{document_id}{file_extension}"
            save_path = Path(settings.data_storage_path) / save_filename

            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(save_path, "wb") as f:
                await f.write(file_content)

        except Exception as e:
            logger.error(f"Failed to save file {filename}: {e}")
            raise e

    def get_processing_status(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get real-time processing status for a file"""
        if file_id not in self.processing_status:
            return None

        status = self.processing_status[file_id]

        # Calculate timing information
        current_time = datetime.now()
        if status.start_time:
            status.elapsed_time = (current_time - status.start_time).total_seconds()

        # Estimate remaining time based on progress
        if status.overall_progress > 0 and status.overall_progress < 100:
            total_estimated = status.elapsed_time / (status.overall_progress / 100)
            status.remaining_time = max(0, total_estimated - status.elapsed_time)

        return {
            "file_id": status.file_id,
            "filename": status.filename,
            "overall_progress": status.overall_progress,
            "status": status.status,
            "elapsed_time": status.elapsed_time,
            "remaining_time": status.remaining_time,
            "estimated_total_time": status.estimated_total_time,
            "parallel_workers": status.parallel_workers,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "progress": chunk.progress,
                    "status": chunk.status,
                    "pages": f"{chunk.start_page}-{chunk.end_page}",
                    "chunks_created": chunk.chunks_created,
                    "error": chunk.error_message,
                }
                for chunk in status.chunks
            ],
        }

    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, "process_pool"):
                self.process_pool.shutdown(wait=True)
                logger.info("Process pool shut down successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


# Global instance
parallel_pdf_processor = ParallelPDFProcessor()
