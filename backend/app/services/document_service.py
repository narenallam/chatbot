"""
Enhanced Document Processing Service for the Personal Assistant AI Chatbot
Handles document upload, parsing, OCR, and storage for specific file types
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
import hashlib
import json
import tempfile

# Document processing imports
import PyPDF2
from docx import Document as DocxDocument

# Additional document processing imports
try:
    from pptx import Presentation

    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

try:
    import pandas as pd
    import openpyxl

    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# Image and OCR processing imports
try:
    from PIL import Image

    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False

# HEIF support
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

# OCR support
try:
    import pytesseract

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Enhanced PDF processing
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# LangChain imports for text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Internal imports
from app.core.config import settings
from app.services.vector_service import vector_service
from app.services.database_service import database_service

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
    """Enhanced document processing and storage service with OCR support"""

    def __init__(self):
        self.db_service = database_service
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        self._setup_storage_directories()
        self._log_capabilities()

    def _setup_storage_directories(self):
        """Setup storage directories"""
        directories = [
            Path("data/hashed_files"),
            Path("data/original_files"),  # Store original files with original names
            Path("data/metadata"),
            Path("data/logs"),
            Path("data/temp"),
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _log_capabilities(self):
        """Log available processing capabilities"""
        capabilities = []
        if IMAGE_AVAILABLE:
            capabilities.append("Image processing (PIL)")
        if HEIF_AVAILABLE:
            capabilities.append("HEIF support")
        if OCR_AVAILABLE:
            capabilities.append("OCR (pytesseract)")
        if PYMUPDF_AVAILABLE:
            capabilities.append("Enhanced PDF (PyMuPDF)")
        if PPTX_AVAILABLE:
            capabilities.append("PowerPoint processing")
        if EXCEL_AVAILABLE:
            capabilities.append("Excel processing")

        logger.info(f"Document processing capabilities: {', '.join(capabilities)}")

    def _calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()

    def _calculate_data_hash(self, file_content: bytes, file_size: int) -> str:
        """Calculate hash of file data + size for duplicate detection"""
        data = file_content + str(file_size).encode()
        return hashlib.sha256(data).hexdigest()

    def _generate_new_filename(self, file_hash: str, original_filename: str) -> str:
        """Generate new filename using hash and original extension"""
        extension = Path(original_filename).suffix.lower()
        return f"{file_hash}{extension}"

    def _get_content_type(self, filename: str) -> str:
        """Get MIME content type from filename"""
        extension = Path(filename).suffix.lower()
        content_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".heic": "image/heic",
            ".heif": "image/heif",
        }
        return content_types.get(extension, "application/octet-stream")

    def _is_supported_file_type(self, filename: str) -> bool:
        """Check if file type is supported"""
        supported_extensions = {
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".png",
            ".jpg",
            ".jpeg",
            ".heic",
            ".heif",
        }
        extension = Path(filename).suffix.lower()
        return extension in supported_extensions

    async def process_uploaded_file(
        self, file_content: bytes, filename: str, content_type: str
    ) -> Dict[str, Any]:
        """
        Process uploaded file with enhanced OCR capabilities

        Args:
            file_content: File content as bytes
            filename: Original filename
            content_type: MIME content type

        Returns:
            Processing result dictionary
        """
        start_time = datetime.now()

        try:
            # Validate file type
            if not self._is_supported_file_type(filename):
                return {
                    "status": "error",
                    "message": f"Unsupported file type: {Path(filename).suffix}",
                    "filename": filename,
                }

            # Calculate hashes
            file_hash = self._calculate_file_hash(file_content)
            file_size = len(file_content)
            file_data_hash = self._calculate_data_hash(file_content, file_size)

            # Check for duplicates
            existing_file = self.db_service.check_file_exists(file_data_hash)
            if existing_file:
                return {
                    "status": "success",
                    "message": "File already exists",
                    "filename": filename,
                    "file_id": existing_file["id"],
                    "duplicate": True,
                }

            # Generate new filename
            new_filename = self._generate_new_filename(file_hash, filename)

            # Save file to hashed_files directory
            file_path = Path("data/hashed_files") / new_filename
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(file_content)

            # Also save original file with original name in original_files directory
            # Generate a unique original filename to handle duplicates
            original_file_path = Path("data/original_files") / filename
            counter = 1
            while original_file_path.exists():
                name_parts = Path(filename).stem, Path(filename).suffix
                original_file_path = (
                    Path("data/original_files")
                    / f"{name_parts[0]}_{counter}{name_parts[1]}"
                )
                counter += 1

            async with aiofiles.open(original_file_path, "wb") as f:
                await f.write(file_content)

            # Extract text content with OCR support
            text_content = await self._extract_text(
                file_content, filename, content_type
            )

            if not text_content.strip():
                return {
                    "status": "error",
                    "message": "No text content could be extracted from the file",
                    "filename": filename,
                }

            # Generate file ID first
            file_id = str(uuid.uuid4())

            # Create chunks for vector storage
            chunks = self._create_chunks(text_content, filename, file_hash, file_id)

            # Store in vector database
            chunk_ids = await vector_service.add_documents(chunks)

            # Save file info to database with the pre-generated file_id
            saved_file_id = self.db_service.save_file_info(
                file_id=file_id,
                full_filename=filename,
                file_hash=file_hash,
                new_filename=new_filename,
                file_size=file_size,
                file_data_hash=file_data_hash,
                content_type=content_type,
                metadata={
                    "chunk_count": len(chunks),
                    "chunk_ids": chunk_ids,
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "ocr_used": self._was_ocr_used(filename, text_content),
                    "original_file_path": str(original_file_path),
                    "hashed_file_path": str(file_path),
                },
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            await log_to_websocket(
                "info",
                f"✅ Successfully processed {filename}",
                {
                    "chunks": len(chunks),
                    "processing_time": processing_time,
                    "file_size_mb": file_size / 1024 / 1024,
                    "ocr_used": self._was_ocr_used(filename, text_content),
                },
            )

            return {
                "status": "success",
                "message": "File processed successfully",
                "filename": filename,
                "file_id": file_id,
                "chunk_count": len(chunks),
                "processing_time_seconds": processing_time,
                "file_size_mb": file_size / 1024 / 1024,
                "new_filename": new_filename,
                "ocr_used": self._was_ocr_used(filename, text_content),
            }

        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            await log_to_websocket("error", f"❌ Error processing {filename}: {str(e)}")

            return {
                "status": "error",
                "message": f"Processing failed: {str(e)}",
                "filename": filename,
            }

    def _was_ocr_used(self, filename: str, text_content: str) -> bool:
        """Check if OCR was likely used based on filename and content"""
        extension = Path(filename).suffix.lower()
        is_image = extension in [".png", ".jpg", ".jpeg", ".heic", ".heif"]

        # If it's an image and has substantial text content, OCR was likely used
        if is_image and len(text_content.strip()) > 50:
            return True

        # Check for OCR indicators in text
        ocr_indicators = ["[OCR]", "scanned", "image text", "extracted from image"]
        return any(
            indicator.lower() in text_content.lower() for indicator in ocr_indicators
        )

    async def _extract_text(
        self, file_content: bytes, filename: str, content_type: str
    ) -> str:
        """
        Extract text from file based on type with OCR support

        Args:
            file_content: File content as bytes
            filename: Original filename
            content_type: MIME content type

        Returns:
            Extracted text content
        """
        file_extension = Path(filename).suffix.lower()

        try:
            if file_extension == ".pdf":
                return await self._extract_from_pdf(file_content)
            elif file_extension == ".docx":
                return await self._extract_from_docx(file_content)
            elif file_extension == ".pptx":
                return await self._extract_from_pptx(file_content)
            elif file_extension == ".xlsx":
                return await self._extract_from_excel(file_content)
            elif file_extension in [".png", ".jpg", ".jpeg", ".heic", ".heif"]:
                return await self._extract_from_image(file_content, filename)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {str(e)}")
            raise

    async def _extract_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file with OCR support for scanned pages"""
        try:
            text_content = []

            # Try PyMuPDF first for better handling
            if PYMUPDF_AVAILABLE:
                try:
                    doc = fitz.open(stream=file_content, filetype="pdf")
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)

                        # Try to extract text normally first
                        page_text = page.get_text()

                        # If no text found and OCR is available, try OCR
                        if not page_text.strip() and OCR_AVAILABLE:
                            await log_to_websocket(
                                "info", f"🔍 Using OCR for page {page_num + 1}"
                            )

                            # Convert page to image
                            pix = page.get_pixmap()
                            img_data = pix.tobytes("png")

                            # Perform OCR on the image
                            img = Image.open(io.BytesIO(img_data))
                            page_text = pytesseract.image_to_string(img)

                            if page_text.strip():
                                page_text = f"[OCR Page {page_num + 1}]\n{page_text}"

                        if page_text.strip():
                            text_content.append(f"[Page {page_num + 1}]\n{page_text}")

                    doc.close()

                    if text_content:
                        return "\n\n".join(text_content)

                except Exception as e:
                    logger.warning(
                        f"PyMuPDF extraction failed: {e}, falling back to PyPDF2"
                    )

            # Fallback to PyPDF2
            with suppress_pdf_warnings():
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                    except Exception as e:
                        logger.warning(
                            f"Error extracting text from PDF page {page_num + 1}: {e}"
                        )
                        continue

                return "\n\n".join(text_content)

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    async def _extract_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file including embedded images"""
        try:
            doc = DocxDocument(io.BytesIO(file_content))
            text_content = []

            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)

            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_content.append(" | ".join(row_text))

            # Extract text from embedded images if OCR is available
            if OCR_AVAILABLE:
                image_text = await self._extract_from_docx_images(doc)
                if image_text:
                    text_content.append("\n=== Embedded Images ===\n")
                    text_content.append(image_text)

            return "\n".join(text_content)

        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            raise

    async def _extract_from_docx_images(self, doc) -> str:
        """Extract text from images embedded in DOCX using OCR"""
        try:
            image_texts = []

            # Extract images from the document
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        image_data = rel.target_part.blob
                        img = Image.open(io.BytesIO(image_data))

                        # Perform OCR on the image
                        ocr_text = pytesseract.image_to_string(img)
                        if ocr_text.strip():
                            image_texts.append(
                                f"[Embedded Image OCR]\n{ocr_text.strip()}"
                            )

                    except Exception as e:
                        logger.warning(f"Error processing embedded image: {e}")
                        continue

            return "\n\n".join(image_texts)

        except Exception as e:
            logger.warning(f"Error extracting from DOCX images: {e}")
            return ""

    async def _extract_from_pptx(self, file_content: bytes) -> str:
        """Extract text from PPTX file including embedded images"""
        if not PPTX_AVAILABLE:
            raise ImportError("python-pptx is not available")

        try:
            prs = Presentation(io.BytesIO(file_content))
            text_content = []

            for slide_num, slide in enumerate(prs.slides):
                slide_text = [f"[Slide {slide_num + 1}]"]

                # Extract text from shapes
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text.append(shape.text.strip())

                # Extract text from embedded images if OCR is available
                if OCR_AVAILABLE:
                    image_text = await self._extract_from_pptx_images(slide)
                    if image_text:
                        slide_text.append(
                            f"[Slide {slide_num + 1} Images]\n{image_text}"
                        )

                if len(slide_text) > 1:  # More than just the slide number
                    text_content.append("\n".join(slide_text))

            return "\n\n".join(text_content)

        except Exception as e:
            logger.error(f"Error extracting text from PPTX: {str(e)}")
            raise

    async def _extract_from_pptx_images(self, slide) -> str:
        """Extract text from images embedded in PPTX slide using OCR"""
        try:
            image_texts = []

            for shape in slide.shapes:
                if hasattr(shape, "image"):
                    try:
                        image_data = shape.image.blob
                        img = Image.open(io.BytesIO(image_data))

                        # Perform OCR on the image
                        ocr_text = pytesseract.image_to_string(img)
                        if ocr_text.strip():
                            image_texts.append(f"[Slide Image OCR]\n{ocr_text.strip()}")

                    except Exception as e:
                        logger.warning(f"Error processing slide image: {e}")
                        continue

            return "\n\n".join(image_texts)

        except Exception as e:
            logger.warning(f"Error extracting from PPTX images: {e}")
            return ""

    async def _extract_from_excel(self, file_content: bytes) -> str:
        """Extract text from Excel file"""
        if not EXCEL_AVAILABLE:
            raise ImportError("pandas/openpyxl is not available")

        try:
            # Try to read with pandas
            df_dict = pd.read_excel(io.BytesIO(file_content), sheet_name=None)
            text_content = []

            for sheet_name, df in df_dict.items():
                if not df.empty:
                    sheet_text = [f"[Sheet: {sheet_name}]"]

                    # Add column headers
                    if not df.columns.empty:
                        sheet_text.append(
                            "Headers: " + " | ".join(str(col) for col in df.columns)
                        )

                    # Add first few rows as sample
                    for idx, row in df.head(10).iterrows():
                        row_text = " | ".join(
                            str(val) for val in row.values if pd.notna(val)
                        )
                        if row_text.strip():
                            sheet_text.append(f"Row {idx + 1}: {row_text}")

                    text_content.append("\n".join(sheet_text))

            return "\n\n".join(text_content)

        except Exception as e:
            logger.error(f"Error extracting text from Excel: {str(e)}")
            raise

    async def _extract_from_image(self, file_content: bytes, filename: str) -> str:
        """Extract text from image file using OCR"""
        if not IMAGE_AVAILABLE:
            raise ImportError("Pillow is not available")

        try:
            # Open image (supports HEIC if pillow-heif is available)
            image = Image.open(io.BytesIO(file_content))

            # Convert to RGB if necessary for OCR
            if image.mode not in ["RGB", "L"]:
                image = image.convert("RGB")

            # Extract basic image metadata
            metadata = [
                f"[Image: {filename}]",
                f"Format: {image.format}",
                f"Size: {image.size[0]}x{image.size[1]} pixels",
                f"Mode: {image.mode}",
            ]

            # Try to extract EXIF data if available
            if hasattr(image, "_getexif") and image._getexif():
                exif = image._getexif()
                if exif:
                    metadata.append("EXIF data available")

            # Perform OCR if available
            ocr_text = ""
            if OCR_AVAILABLE:
                try:
                    await log_to_websocket("info", f"🔍 Performing OCR on {filename}")
                    ocr_text = pytesseract.image_to_string(image)

                    if ocr_text.strip():
                        ocr_text = f"\n=== OCR Extracted Text ===\n{ocr_text.strip()}"
                    else:
                        ocr_text = "\n=== OCR: No text detected ==="

                except Exception as e:
                    logger.warning(f"OCR failed for {filename}: {e}")
                    ocr_text = f"\n=== OCR failed: {str(e)} ==="
            else:
                ocr_text = "\n=== OCR not available ==="

            return "\n".join(metadata) + ocr_text

        except Exception as e:
            logger.error(f"Error extracting metadata from image: {str(e)}")
            raise

    def _create_chunks(
        self, text_content: str, filename: str, file_hash: str, document_id: str
    ) -> List[Document]:
        """
        Create text chunks for vector storage

        Args:
            text_content: Full text content
            filename: Original filename
            file_hash: File hash for reference

        Returns:
            List of Document chunks
        """
        try:
            chunks = self.text_splitter.split_text(text_content)

            documents = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "file_hash": file_hash,
                            "document_id": document_id,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "chunk_id": f"{file_hash}_chunk_{i}",
                        },
                    )
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error creating chunks for {filename}: {str(e)}")
            raise

    async def get_document_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get document information by file ID"""
        try:
            files = self.db_service.get_files()
            for file in files:
                if file["id"] == file_id:
                    # Add file path information for preview
                    file_info = file.copy()

                    # Determine the file path based on storage structure
                    original_path = Path("data/original_files") / file["full_filename"]
                    hashed_path = Path("data/hashed_files") / file["new_filename"]

                    # Prefer original file if it exists, otherwise use hashed file
                    if original_path.exists():
                        file_info["file_path"] = str(original_path)
                    elif hashed_path.exists():
                        file_info["file_path"] = str(hashed_path)
                    else:
                        logger.warning(
                            f"File not found for document {file_id}: {file['full_filename']}"
                        )
                        file_info["file_path"] = None

                    # Add metadata for preview endpoint compatibility
                    if "metadata" not in file_info:
                        file_info["metadata"] = {}

                    file_info["metadata"]["filename"] = file["full_filename"]
                    file_info["metadata"]["file_path"] = file_info["file_path"]
                    file_info["metadata"]["content_type"] = file["content_type"]

                    return file_info
            return None
        except Exception as e:
            logger.error(f"Error getting document info: {str(e)}")
            return None

    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all uploaded documents"""
        try:
            return self.db_service.get_files()
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []

    async def delete_document(self, file_id: str) -> bool:
        """Delete a document"""
        try:
            # Get file info
            file_info = await self.get_document_info(file_id)
            if not file_info:
                return False

            # Delete from vector database
            if "chunk_ids" in file_info.get("metadata", {}):
                chunk_ids = file_info["metadata"]["chunk_ids"]
                await vector_service.delete_documents(chunk_ids)

            # Delete file from storage
            file_path = Path("data/hashed_files") / file_info["new_filename"]
            if file_path.exists():
                file_path.unlink()

            # Delete from database
            return self.db_service.delete_file(file_id)

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    async def search_documents(
        self, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search documents using vector similarity"""
        try:
            results = await vector_service.search(query, limit)
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []


# Create global instance
document_service = DocumentService()
