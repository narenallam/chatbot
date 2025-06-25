"""
Document Processing Service for the Personal Assistant AI Chatbot
Handles document upload, parsing, chunking, and vector storage
"""

import asyncio
import aiofiles
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import warnings
import io
import sys
from contextlib import contextmanager

# Document processing imports
import PyPDF2
from docx import Document as DocxDocument
from bs4 import BeautifulSoup

# LangChain imports for text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Internal imports
from app.core.config import settings
from app.services.vector_service import vector_service
from app.services.database_service import DatabaseService

logger = logging.getLogger(__name__)

# Global function to broadcast logs (will be set by main.py)
broadcast_log_func = None


async def log_to_websocket(level: str, message: str, details: dict = None):
    """Send log message to WebSocket clients"""
    if broadcast_log_func:
        try:
            await broadcast_log_func(level, message, details)
        except Exception as e:
            logger.error(f"Failed to broadcast log: {e}")


@contextmanager
def suppress_pdf_warnings():
    """Context manager to suppress PyPDF2 warnings and capture them"""
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            yield
        finally:
            captured = sys.stderr.getvalue()
            sys.stderr = old_stderr
            if captured:
                logger.debug(f"PDF parsing messages: {captured[:200]}...")


class DocumentService:
    """Document processing and storage service"""

    def __init__(self):
        self.db_service = DatabaseService()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )

    async def process_uploaded_file(
        self, file_content: bytes, filename: str, content_type: str
    ) -> Dict[str, Any]:
        """
        Process an uploaded file and store it in the vector database

        Args:
            file_content: File content as bytes
            filename: Original filename
            content_type: MIME type of the file

        Returns:
            Processing result with document ID and statistics
        """
        document_id = str(uuid.uuid4())
        start_time = datetime.now()

        # Initialize status tracking
        status = {
            "document_id": document_id,
            "filename": filename,
            "file_size": len(file_content),
            "status": "processing",
            "stages": {
                "extraction": {
                    "status": "starting",
                    "message": "Preparing to extract text",
                },
                "chunking": {
                    "status": "pending",
                    "message": "Waiting for text extraction",
                },
                "embedding": {"status": "pending", "message": "Waiting for chunking"},
                "storage": {"status": "pending", "message": "Waiting for embeddings"},
            },
        }

        try:
            # Stage 1: Text Extraction
            logger.info(f"Starting text extraction for {filename}")
            status["stages"]["extraction"]["status"] = "processing"
            status["stages"]["extraction"][
                "message"
            ] = f"Extracting text from {self._get_document_type(filename).upper()} file"

            await log_to_websocket(
                "info",
                f"🔍 Starting text extraction from {self._get_document_type(filename).upper()}",
            )
            text_content = await self._extract_text(
                file_content, filename, content_type
            )
            await log_to_websocket(
                "success",
                f"📄 Extracted {len(text_content)} characters from {filename}",
            )

            if not text_content.strip():
                status["stages"]["extraction"]["status"] = "failed"
                status["stages"]["extraction"][
                    "message"
                ] = "No readable text found in file"
                raise ValueError("No text content could be extracted from the file")

            status["stages"]["extraction"]["status"] = "completed"
            status["stages"]["extraction"][
                "message"
            ] = f"Extracted {len(text_content)} characters"

            # Stage 2: Text Chunking
            logger.info(f"Starting text chunking for {filename}")
            status["stages"]["chunking"]["status"] = "processing"
            status["stages"]["chunking"][
                "message"
            ] = "Splitting text into semantic chunks"

            await log_to_websocket("info", f"✂️ Creating semantic chunks from text")
            chunks = self._create_chunks(text_content, filename, document_id)
            await log_to_websocket("success", f"📝 Created {len(chunks)} text chunks")

            if not chunks:
                status["stages"]["chunking"]["status"] = "failed"
                status["stages"]["chunking"][
                    "message"
                ] = "Failed to create chunks from text"
                raise ValueError("No chunks could be created from the document")

            status["stages"]["chunking"]["status"] = "completed"
            status["stages"]["chunking"]["message"] = f"Created {len(chunks)} chunks"

            # Stage 3: Embedding Generation & Vector Storage
            logger.info(f"Starting embedding generation for {filename}")
            status["stages"]["embedding"]["status"] = "processing"
            status["stages"]["embedding"]["message"] = "Generating vector embeddings"

            await log_to_websocket(
                "info", f"🧠 Generating AI embeddings for {len(chunks)} chunks"
            )
            chunk_ids = vector_service.add_documents(
                texts=[chunk.page_content for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                document_id=document_id,
            )
            await log_to_websocket(
                "success", f"⚡ Generated embeddings and stored in vector database"
            )

            status["stages"]["embedding"]["status"] = "completed"
            status["stages"]["embedding"][
                "message"
            ] = f"Generated embeddings for {len(chunk_ids)} chunks"

            # Stage 4: Database Storage
            logger.info(f"Storing metadata for {filename}")
            status["stages"]["storage"]["status"] = "processing"
            status["stages"]["storage"]["message"] = "Saving document metadata"

            await log_to_websocket("info", f"💾 Saving document metadata to database")
            self.db_service.save_document(
                filename=filename,
                content=(
                    text_content[:1000] + "..."
                    if len(text_content) > 1000
                    else text_content
                ),
                doc_type=self._get_document_type(filename),
                metadata={
                    "content_type": content_type,
                    "file_size": len(file_content),
                    "chunk_count": len(chunks),
                    "processing_date": datetime.now().isoformat(),
                    "text_length": len(text_content),
                },
            )

            # Save full content to file
            await self._save_file(file_content, document_id, filename)

            status["stages"]["storage"]["status"] = "completed"
            status["stages"]["storage"]["message"] = "Document metadata saved"

            # Final status
            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Successfully processed {filename} in {processing_time:.2f}s with {len(chunks)} chunks"
            )

            return {
                "document_id": document_id,
                "filename": filename,
                "chunk_count": len(chunks),
                "text_length": len(text_content),
                "file_size": len(file_content),
                "processing_time_seconds": round(processing_time, 2),
                "status": "success",
                "message": f"Document processed successfully with {len(chunks)} chunks in {processing_time:.1f}s",
                "stages": status["stages"],
            }

        except Exception as e:
            # Update status with error information
            processing_time = (datetime.now() - start_time).total_seconds()
            error_message = str(e)

            logger.error(f"Error processing document {filename}: {error_message}")

            # Determine which stage failed
            failed_stage = "extraction"
            for stage_name, stage_info in status["stages"].items():
                if stage_info["status"] == "processing":
                    failed_stage = stage_name
                    break
                elif stage_info["status"] == "failed":
                    failed_stage = stage_name
                    break

            return {
                "document_id": document_id,
                "filename": filename,
                "file_size": len(file_content),
                "processing_time_seconds": round(processing_time, 2),
                "status": "error",
                "message": f"Failed during {failed_stage}: {error_message}",
                "error_stage": failed_stage,
                "error_details": error_message,
                "stages": status["stages"],
            }

    async def _extract_text(
        self, file_content: bytes, filename: str, content_type: str
    ) -> str:
        """Extract text from different file types"""
        try:
            file_extension = Path(filename).suffix.lower()

            if file_extension == ".pdf" or "pdf" in content_type:
                return await self._extract_from_pdf(file_content)
            elif file_extension in [".docx", ".doc"] or "word" in content_type:
                return await self._extract_from_docx(file_content)
            elif file_extension in [".txt", ".md"] or "text" in content_type:
                return await self._extract_from_text(file_content)
            elif file_extension in [".html", ".htm"]:
                return await self._extract_from_html(file_content)
            else:
                # Try as text file as fallback
                return await self._extract_from_text(file_content)

        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {e}")
            raise

    async def _extract_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file with better error handling"""
        try:
            logger.info("Processing PDF file...")
            await log_to_websocket(
                "info", "📖 Opening PDF file and analyzing structure"
            )

            with suppress_pdf_warnings():
                pdf_file = io.BytesIO(file_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                total_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {total_pages} pages")
                await log_to_websocket("info", f"📄 Found {total_pages} pages in PDF")

                text_content = []
                successful_pages = 0

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        if (
                            page_num % 5 == 0 or page_num == 0
                        ):  # Log every 5 pages or first page
                            await log_to_websocket(
                                "info",
                                f"📃 Processing page {page_num + 1}/{total_pages}",
                            )

                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                            successful_pages += 1
                    except Exception as page_error:
                        logger.warning(
                            f"Could not extract text from page {page_num + 1}: {page_error}"
                        )
                        await log_to_websocket(
                            "warning", f"⚠️ Skipped page {page_num + 1} (unreadable)"
                        )
                        continue

                logger.info(
                    f"Successfully extracted text from {successful_pages}/{total_pages} pages"
                )
                await log_to_websocket(
                    "success",
                    f"✅ Extracted text from {successful_pages}/{total_pages} pages",
                )

                if not text_content:
                    raise ValueError(
                        f"Could not extract readable text from any of the {total_pages} pages"
                    )

                return "\n\n".join(text_content)

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            if "encrypted" in str(e).lower():
                raise ValueError("PDF file is encrypted or password-protected")
            elif "corrupt" in str(e).lower():
                raise ValueError("PDF file appears to be corrupted")
            else:
                raise ValueError(f"Failed to extract text from PDF: {str(e)[:100]}")

    async def _extract_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            docx_file = io.BytesIO(file_content)
            doc = DocxDocument(docx_file)

            text_content = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)

            return "\n\n".join(text_content)
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise ValueError(f"Failed to extract text from DOCX: {e}")

    async def _extract_from_text(self, file_content: bytes) -> str:
        """Extract text from plain text file"""
        try:
            # Try different encodings
            for encoding in ["utf-8", "utf-16", "latin-1", "cp1252"]:
                try:
                    return file_content.decode(encoding)
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, use utf-8 with error handling
            return file_content.decode("utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise ValueError(f"Failed to extract text: {e}")

    async def _extract_from_html(self, file_content: bytes) -> str:
        """Extract text from HTML file"""
        try:
            html_content = file_content.decode("utf-8", errors="replace")
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text and clean up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            return text
        except Exception as e:
            logger.error(f"Error extracting HTML text: {e}")
            raise ValueError(f"Failed to extract text from HTML: {e}")

    def _create_chunks(
        self, text_content: str, filename: str, document_id: str
    ) -> List[Document]:
        """Create document chunks for vector storage"""
        try:
            # Create a LangChain Document
            doc = Document(
                page_content=text_content,
                metadata={
                    "filename": filename,
                    "document_id": document_id,
                    "doc_type": self._get_document_type(filename),
                },
            )

            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])

            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update(
                    {
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_id": f"{document_id}_chunk_{i}",
                    }
                )

            return chunks
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            raise

    def _get_document_type(self, filename: str) -> str:
        """Determine document type from filename"""
        extension = Path(filename).suffix.lower()
        type_mapping = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".doc": "doc",
            ".txt": "text",
            ".md": "markdown",
            ".html": "html",
            ".htm": "html",
        }
        return type_mapping.get(extension, "unknown")

    async def _save_file(self, file_content: bytes, document_id: str, filename: str):
        """Save the original file to disk"""
        try:
            # Create data directory
            data_dir = Path(settings.data_storage_path)
            data_dir.mkdir(parents=True, exist_ok=True)

            # Save with document ID as filename to avoid conflicts
            file_extension = Path(filename).suffix
            save_path = data_dir / f"{document_id}{file_extension}"

            async with aiofiles.open(save_path, "wb") as f:
                await f.write(file_content)

            logger.info(f"Saved file {filename} to {save_path}")
        except Exception as e:
            logger.error(f"Error saving file {filename}: {e}")
            # Don't raise here as the document processing was successful

    async def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a processed document"""
        try:
            # Get document metadata from database
            documents = self.db_service.get_documents()
            document = next(
                (doc for doc in documents if doc["id"] == document_id), None
            )

            if not document:
                return None

            # Get chunks from vector database
            chunks = vector_service.get_document_chunks(document_id)

            return {
                "document_id": document_id,
                "filename": document["filename"],
                "doc_type": document["doc_type"],
                "upload_date": document["upload_date"],
                "chunk_count": len(chunks),
                "chunks": chunks[:3] if chunks else [],  # First 3 chunks as preview
                "metadata": document.get("metadata", {}),
            }
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return None

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks"""
        try:
            # Delete from vector database
            success = vector_service.delete_document(document_id)

            if success:
                # Delete file from disk
                data_dir = Path(settings.data_storage_path)
                for file_path in data_dir.glob(f"{document_id}.*"):
                    file_path.unlink()

                logger.info(f"Deleted document {document_id}")
                return True

            return False
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents"""
        try:
            documents = self.db_service.get_documents()

            # Add chunk count for each document
            for doc in documents:
                chunks = vector_service.get_document_chunks(doc["id"])
                doc["chunk_count"] = len(chunks)

            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []


# Global document service instance
document_service = DocumentService()
