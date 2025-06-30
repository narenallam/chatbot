"""
Multiprocessing Service for True CPU Parallelism

This service implements real multiprocessing for CPU-intensive tasks while properly handling:
1. HuggingFace tokenizer parallelism conflicts
2. Process pool management
3. Memory-efficient chunk processing
4. Safe inter-process communication

Key Design Principles:
- Separate process pools for different task types
- Proper tokenizer initialization per process
- Memory management for large files
- Error handling and process recovery
"""

import os
import sys
import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue, Process
from concurrent.futures import ProcessPoolExecutor, as_completed
import asyncio
from functools import partial
import logging
import time
import pickle
import traceback
from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path
import threading
from queue import Queue as ThreadQueue
import signal

# Fix HuggingFace tokenizers parallelism BEFORE any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Now import ML libraries
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

# Import document processing libraries
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from app.core.config import settings

logger = logging.getLogger(__name__)


class ProcessSafeEmbeddingModel:
    """Process-safe embedding model that handles tokenizer parallelism"""

    _model_cache = {}
    _lock = threading.Lock()

    @classmethod
    def get_model(cls, model_name: str):
        """Get or create embedding model in a process-safe way"""
        process_id = os.getpid()
        cache_key = f"{model_name}_{process_id}"

        with cls._lock:
            if cache_key not in cls._model_cache:
                # Set tokenizers parallelism to false for this process
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                # Initialize model in this process
                logger.info(f"Initializing embedding model in process {process_id}")
                model = SentenceTransformer(model_name)
                cls._model_cache[cache_key] = model

            return cls._model_cache[cache_key]


def init_worker_process():
    """Initialize worker process with proper environment settings"""
    # Set environment variables for the worker process
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"  # Prevent nested parallelism
    os.environ["MKL_NUM_THREADS"] = "1"

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    # Initialize logging for the worker process
    logging.basicConfig(level=logging.INFO)


def process_pdf_pages_worker(args: Tuple) -> Dict[str, Any]:
    """Worker function to process PDF pages in parallel"""
    try:
        file_content, page_range, filename = args

        if not PYMUPDF_AVAILABLE:
            return {"error": "PyMuPDF not available", "pages": []}

        import fitz
        import io

        doc = fitz.open(stream=file_content, filetype="pdf")
        pages_text = []

        start_page, end_page = page_range

        for page_num in range(start_page, min(end_page, len(doc))):
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text()

                # If no text and OCR available, try OCR
                if not page_text.strip() and OCR_AVAILABLE:
                    try:
                        pix = page.get_pixmap()
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        page_text = pytesseract.image_to_string(img)
                        if page_text.strip():
                            page_text = f"[OCR Page {page_num + 1}]\\n{page_text}"
                    except Exception as ocr_e:
                        logger.warning(f"OCR failed for page {page_num + 1}: {ocr_e}")

                if page_text.strip():
                    pages_text.append(
                        {
                            "page_num": page_num + 1,
                            "text": f"[Page {page_num + 1}]\\n{page_text}",
                        }
                    )

            except Exception as e:
                logger.warning(f"Error processing page {page_num + 1}: {e}")
                continue

        doc.close()

        return {
            "success": True,
            "pages": pages_text,
            "page_range": page_range,
            "filename": filename,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "page_range": page_range,
            "filename": filename,
        }


def generate_embeddings_worker(args: Tuple) -> Dict[str, Any]:
    """Worker function to generate embeddings in parallel"""
    try:
        texts, model_name, chunk_batch_id = args

        # Get process-safe model
        model = ProcessSafeEmbeddingModel.get_model(model_name)

        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False)

        return {
            "success": True,
            "embeddings": embeddings.tolist(),
            "chunk_batch_id": chunk_batch_id,
            "texts_count": len(texts),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "chunk_batch_id": chunk_batch_id,
        }


def process_ocr_images_worker(args: Tuple) -> Dict[str, Any]:
    """Worker function to process OCR on images in parallel"""
    try:
        image_data_list, filename = args

        if not OCR_AVAILABLE:
            return {"error": "OCR not available", "texts": []}

        import pytesseract
        from PIL import Image
        import io

        ocr_results = []

        for i, image_data in enumerate(image_data_list):
            try:
                img = Image.open(io.BytesIO(image_data))
                ocr_text = pytesseract.image_to_string(img, config="--psm 6")

                if ocr_text.strip():
                    ocr_results.append(
                        {
                            "image_index": i,
                            "text": f"[Image {i + 1} OCR]\\n{ocr_text.strip()}",
                        }
                    )
            except Exception as e:
                logger.warning(f"OCR failed for image {i + 1}: {e}")
                continue

        return {"success": True, "texts": ocr_results, "filename": filename}

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "filename": filename,
        }


class MultiprocessingService:
    """Service for managing multiprocessing tasks in document processing"""

    def __init__(self):
        self.max_workers = min(mp.cpu_count(), settings.max_workers)
        self.embedding_batch_size = 32  # Process embeddings in batches
        self.pdf_page_batch_size = 10  # Process PDF pages in batches

        # Different pools for different task types
        self._pdf_pool = None
        self._embedding_pool = None
        self._ocr_pool = None

        logger.info(
            f"MultiprocessingService initialized with {self.max_workers} workers"
        )

    def _get_pdf_pool(self):
        """Get or create PDF processing pool"""
        if self._pdf_pool is None:
            self._pdf_pool = ProcessPoolExecutor(
                max_workers=min(4, self.max_workers),  # Limit PDF workers
                initializer=init_worker_process,
            )
        return self._pdf_pool

    def _get_embedding_pool(self):
        """Get or create embedding processing pool"""
        if self._embedding_pool is None:
            self._embedding_pool = ProcessPoolExecutor(
                max_workers=min(
                    2, self.max_workers
                ),  # Limit embedding workers due to memory
                initializer=init_worker_process,
            )
        return self._embedding_pool

    def _get_ocr_pool(self):
        """Get or create OCR processing pool"""
        if self._ocr_pool is None:
            self._ocr_pool = ProcessPoolExecutor(
                max_workers=min(3, self.max_workers),  # Moderate OCR workers
                initializer=init_worker_process,
            )
        return self._ocr_pool

    async def process_pdf_parallel(self, file_content: bytes, filename: str) -> str:
        """Process PDF pages in parallel using multiprocessing"""
        try:
            if not PYMUPDF_AVAILABLE:
                raise ImportError("PyMuPDF not available for parallel PDF processing")

            import fitz
            import io

            # Open document to get page count
            doc = fitz.open(stream=file_content, filetype="pdf")
            total_pages = len(doc)
            doc.close()

            if total_pages <= 5:
                # For small PDFs, don't use multiprocessing overhead
                return await self._process_pdf_sequential(file_content, filename)

            logger.info(f"Processing {total_pages} pages of {filename} in parallel")

            # Split pages into batches for parallel processing
            page_batches = []
            for i in range(0, total_pages, self.pdf_page_batch_size):
                end_page = min(i + self.pdf_page_batch_size, total_pages)
                page_batches.append((i, end_page))

            # Prepare arguments for worker processes
            worker_args = [
                (file_content, page_range, filename) for page_range in page_batches
            ]

            # Process batches in parallel
            loop = asyncio.get_event_loop()
            pool = self._get_pdf_pool()

            # Submit tasks to process pool
            futures = [
                loop.run_in_executor(pool, process_pdf_pages_worker, args)
                for args in worker_args
            ]

            # Wait for all tasks to complete
            results = await asyncio.gather(*futures, return_exceptions=True)

            # Combine results
            all_pages = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"PDF batch processing error: {result}")
                    continue

                if result.get("success"):
                    all_pages.extend(result.get("pages", []))
                else:
                    logger.error(f"PDF batch failed: {result.get('error')}")

            # Sort pages by page number and combine text
            all_pages.sort(key=lambda x: x["page_num"])
            combined_text = "\\n\\n".join([page["text"] for page in all_pages])

            logger.info(
                f"Parallel PDF processing completed: {len(all_pages)} pages extracted"
            )
            return combined_text

        except Exception as e:
            logger.error(f"Parallel PDF processing failed: {e}")
            # Fallback to sequential processing
            return await self._process_pdf_sequential(file_content, filename)

    async def _process_pdf_sequential(self, file_content: bytes, filename: str) -> str:
        """Fallback sequential PDF processing"""
        # This would call the existing PDF processing logic
        # For now, return empty string as placeholder
        logger.info(f"Using sequential PDF processing for {filename}")
        return ""

    async def generate_embeddings_parallel(
        self, texts: List[str], model_name: str
    ) -> List[List[float]]:
        """Generate embeddings in parallel using multiprocessing"""
        try:
            if len(texts) <= 10:
                # For small batches, use sequential processing to avoid overhead
                return await self._generate_embeddings_sequential(texts, model_name)

            logger.info(f"Generating embeddings for {len(texts)} texts in parallel")

            # Split texts into batches for parallel processing
            text_batches = []
            for i in range(0, len(texts), self.embedding_batch_size):
                batch = texts[i : i + self.embedding_batch_size]
                text_batches.append((batch, model_name, i // self.embedding_batch_size))

            # Process batches in parallel
            loop = asyncio.get_event_loop()
            pool = self._get_embedding_pool()

            # Submit tasks to process pool
            futures = [
                loop.run_in_executor(pool, generate_embeddings_worker, args)
                for args in text_batches
            ]

            # Wait for all tasks to complete
            results = await asyncio.gather(*futures, return_exceptions=True)

            # Combine results maintaining order
            all_embeddings = [None] * len(text_batches)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Embedding batch processing error: {result}")
                    continue

                if result.get("success"):
                    batch_id = result.get("chunk_batch_id")
                    all_embeddings[batch_id] = result.get("embeddings", [])
                else:
                    logger.error(f"Embedding batch failed: {result.get('error')}")

            # Flatten results while maintaining order
            final_embeddings = []
            for batch_embeddings in all_embeddings:
                if batch_embeddings:
                    final_embeddings.extend(batch_embeddings)

            logger.info(
                f"Parallel embedding generation completed: {len(final_embeddings)} embeddings"
            )
            return final_embeddings

        except Exception as e:
            logger.error(f"Parallel embedding generation failed: {e}")
            # Fallback to sequential processing
            return await self._generate_embeddings_sequential(texts, model_name)

    async def _generate_embeddings_sequential(
        self, texts: List[str], model_name: str
    ) -> List[List[float]]:
        """Fallback sequential embedding generation"""
        logger.info(f"Using sequential embedding generation for {len(texts)} texts")
        # This would call the existing embedding logic
        # For now, return empty list as placeholder
        return []

    async def process_ocr_images_parallel(
        self, image_data_list: List[bytes], filename: str
    ) -> List[str]:
        """Process OCR on multiple images in parallel"""
        try:
            if not OCR_AVAILABLE:
                return []

            if len(image_data_list) <= 2:
                # For small image sets, use sequential processing
                return await self._process_ocr_sequential(image_data_list, filename)

            logger.info(f"Processing {len(image_data_list)} images for OCR in parallel")

            # Split images into batches for parallel processing
            batch_size = 3  # Process 3 images per batch
            image_batches = []
            for i in range(0, len(image_data_list), batch_size):
                batch = image_data_list[i : i + batch_size]
                image_batches.append((batch, f"{filename}_batch_{i}"))

            # Process batches in parallel
            loop = asyncio.get_event_loop()
            pool = self._get_ocr_pool()

            # Submit tasks to process pool
            futures = [
                loop.run_in_executor(pool, process_ocr_images_worker, args)
                for args in image_batches
            ]

            # Wait for all tasks to complete
            results = await asyncio.gather(*futures, return_exceptions=True)

            # Combine results
            all_texts = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"OCR batch processing error: {result}")
                    continue

                if result.get("success"):
                    texts = result.get("texts", [])
                    all_texts.extend([item["text"] for item in texts])
                else:
                    logger.error(f"OCR batch failed: {result.get('error')}")

            logger.info(
                f"Parallel OCR processing completed: {len(all_texts)} text extractions"
            )
            return all_texts

        except Exception as e:
            logger.error(f"Parallel OCR processing failed: {e}")
            # Fallback to sequential processing
            return await self._process_ocr_sequential(image_data_list, filename)

    async def _process_ocr_sequential(
        self, image_data_list: List[bytes], filename: str
    ) -> List[str]:
        """Fallback sequential OCR processing"""
        logger.info(
            f"Using sequential OCR processing for {len(image_data_list)} images"
        )
        # This would call the existing OCR logic
        # For now, return empty list as placeholder
        return []

    def cleanup(self):
        """Clean up process pools"""
        try:
            if self._pdf_pool:
                self._pdf_pool.shutdown(wait=True)
                self._pdf_pool = None

            if self._embedding_pool:
                self._embedding_pool.shutdown(wait=True)
                self._embedding_pool = None

            if self._ocr_pool:
                self._ocr_pool.shutdown(wait=True)
                self._ocr_pool = None

            logger.info("MultiprocessingService cleaned up successfully")

        except Exception as e:
            logger.error(f"Error during MultiprocessingService cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


# Global service instance
multiprocessing_service = MultiprocessingService()
