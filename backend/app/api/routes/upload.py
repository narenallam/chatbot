"""
Upload API routes for the Personal Assistant AI Chatbot
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Request
from typing import List, Dict, Any, Optional
import logging
import asyncio
from datetime import datetime
from fastapi.responses import JSONResponse
import uuid
import base64

from app.services.document_service import document_service

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


router = APIRouter()

# In-memory storage for upload status (in production, use Redis or database)
upload_status_store = {}

# Store batch processing status in memory (in production, use Redis or database)
batch_status_store: Dict[str, Dict[str, Any]] = {}


@router.post("/documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Unified upload endpoint - intelligently handles single/multiple files with optimal processing

    Features:
    - Single endpoint for all file uploads
    - Intelligent parallel processing based on file characteristics
    - Synchronized WebSocket events across all processes
    - Real-time progress tracking
    """

    # Use asyncio.Lock to prevent WebSocket message race conditions during parallel processing
    websocket_lock = asyncio.Lock()

    async def safe_log_to_websocket(level: str, message: str, details: dict = None):
        """Thread-safe WebSocket logging to prevent message duplication"""
        async with websocket_lock:
            await log_to_websocket(level, message, details)

    try:
        results = []
        upload_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"  # Add microseconds for uniqueness

        logger.info(f"Starting unified upload {upload_id} with {len(files)} files")
        await safe_log_to_websocket(
            "info",
            f"📁 Starting upload batch with {len(files)} files",
            {"upload_id": upload_id, "file_count": len(files)},
        )

        # Analyze files and determine optimal processing strategy
        file_contents = []
        total_size = 0
        large_files_count = 0
        pdf_files_count = 0

        # Phase 1: File validation and reading (sequential for safety)
        for file_idx, file in enumerate(files):
            await safe_log_to_websocket(
                "info",
                f"📄 Analyzing file {file_idx + 1}/{len(files)}: {file.filename}",
                {
                    "file_index": file_idx + 1,
                    "total_files": len(files),
                    "filename": file.filename,
                },
            )

            # Validate file
            if not file.filename:
                results.append(
                    {
                        "filename": "unknown",
                        "status": "error",
                        "message": "No filename provided",
                    }
                )
                continue

            if file.size and file.size > 500 * 1024 * 1024:  # 500MB limit
                results.append(
                    {
                        "filename": file.filename,
                        "status": "error",
                        "message": "File too large (max 500MB)",
                    }
                )
                continue

            try:
                file_content = await file.read()
                file_size = len(file_content)
                total_size += file_size

                # Analyze file characteristics for processing strategy
                is_large = (
                    file_size > 10 * 1024 * 1024
                )  # 10MB+ (lowered for better parallel processing)
                is_pdf = file.filename.lower().endswith(".pdf")

                if is_large:
                    large_files_count += 1
                if is_pdf:
                    pdf_files_count += 1

                file_contents.append(
                    {
                        "file": file,
                        "content": file_content,
                        "file_idx": file_idx,
                        "size": file_size,
                        "is_large": is_large,
                        "is_pdf": is_pdf,
                    }
                )

                await safe_log_to_websocket(
                    "success",
                    f"✅ Prepared {file.filename} ({file_size/1024:.1f} KB)",
                    {
                        "file_size_bytes": file_size,
                        "file_type": "PDF" if is_pdf else "Document",
                    },
                )

            except Exception as read_error:
                await safe_log_to_websocket(
                    "error", f"❌ Failed to read {file.filename}: {str(read_error)}"
                )
                results.append(
                    {
                        "filename": file.filename,
                        "status": "error",
                        "message": f"Failed to read file: {str(read_error)}",
                    }
                )
                continue

        if not file_contents:
            await safe_log_to_websocket("error", "❌ No valid files to process")
            return {
                "upload_id": upload_id,
                "message": "No valid files to process",
                "files": results,
            }

        # Phase 2: Determine processing strategy
        # Use parallel processing more aggressively for better user experience
        use_parallel = (
            len(file_contents) > 1  # Always parallel for multiple files
            and total_size > 5 * 1024 * 1024  # Only if total > 5MB to avoid overhead
        ) or (
            large_files_count > 0  # Always parallel for large files
            or total_size > 50 * 1024 * 1024  # Parallel for 50MB+ total
        )

        processing_strategy = "parallel" if use_parallel else "sequential"
        await safe_log_to_websocket(
            "info",
            f"🧠 Processing strategy: {processing_strategy} | Files: {len(file_contents)} | Large files: {large_files_count} | PDFs: {pdf_files_count} | Total: {total_size/1024/1024:.1f}MB",
            {
                "strategy": processing_strategy,
                "total_files": len(file_contents),
                "large_files": large_files_count,
                "pdf_files": pdf_files_count,
                "total_size_mb": total_size / 1024 / 1024,
            },
        )

        # Phase 3: Process files with synchronized logging
        async def process_single_file_synchronized(file_data):
            """Process a single file with synchronized WebSocket logging"""
            file = file_data["file"]
            file_content = file_data["content"]
            file_idx = file_data["file_idx"]
            file_size = file_data["size"]
            is_large = file_data["is_large"]

            try:
                # Determine service based on file characteristics (not just size)
                service_choice = (
                    "enhanced_ocr"
                    if (is_large or file.filename.lower().endswith(".pdf"))
                    else "standard"
                )

                await safe_log_to_websocket(
                    "info",
                    f"🔍 Processing {file.filename} using {service_choice} service ({file_size/1024/1024:.1f}MB)",
                    {
                        "filename": file.filename,
                        "service": service_choice,
                        "file_size_mb": file_size / 1024 / 1024,
                        "file_index": file_idx + 1,
                        "total_files": len(files),
                    },
                )

                # Use the enhanced document service for all files
                result = await document_service.process_uploaded_file(
                    file_content=file_content,
                    filename=file.filename,
                    content_type=file.content_type or "application/octet-stream",
                )

                # Add processing metadata
                result["upload_id"] = upload_id
                result["file_index"] = file_idx + 1
                result["total_files"] = len(files)
                result["processing_strategy"] = processing_strategy
                result["service_used"] = service_choice

                # Log completion (only errors - success is logged by services)
                if result["status"] != "success":
                    await safe_log_to_websocket(
                        "error",
                        f"❌ Failed to process {file.filename}: {result.get('message', 'Unknown error')}",
                        {"filename": file.filename, "error": result.get("message")},
                    )

                return result

            except Exception as e:
                error_msg = f"Processing failed for {file.filename}: {str(e)}"
                await safe_log_to_websocket("error", f"❌ {error_msg}")
                return {
                    "filename": file.filename,
                    "status": "error",
                    "message": error_msg,
                    "upload_id": upload_id,
                    "file_index": file_idx + 1,
                    "total_files": len(files),
                    "processing_strategy": processing_strategy,
                }

        # Execute processing based on strategy
        if use_parallel:
            await safe_log_to_websocket(
                "info",
                f"🚀 Starting parallel processing of {len(file_contents)} files",
                {"parallel_count": len(file_contents), "strategy": "parallel"},
            )

            # Process files in parallel with synchronized logging
            parallel_results = await asyncio.gather(
                *[
                    process_single_file_synchronized(file_data)
                    for file_data in file_contents
                ],
                return_exceptions=True,
            )

            # Handle results and exceptions
            for result in parallel_results:
                if isinstance(result, Exception):
                    await safe_log_to_websocket(
                        "error", f"❌ Processing exception: {str(result)}"
                    )
                    results.append(
                        {
                            "filename": "unknown",
                            "status": "error",
                            "message": f"Processing exception: {str(result)}",
                        }
                    )
                else:
                    results.append(result)
        else:
            await safe_log_to_websocket(
                "info",
                f"⚡ Starting sequential processing of {len(file_contents)} files",
                {"file_count": len(file_contents), "strategy": "sequential"},
            )

            # Process files sequentially
            for file_data in file_contents:
                result = await process_single_file_synchronized(file_data)
                if isinstance(result, Exception):
                    await safe_log_to_websocket(
                        "error", f"❌ Processing exception: {str(result)}"
                    )
                    results.append(
                        {
                            "filename": "unknown",
                            "status": "error",
                            "message": f"Processing exception: {str(result)}",
                        }
                    )
                else:
                    results.append(result)

        # Count successful and failed uploads
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful

        # Store status for tracking
        upload_status_store[upload_id] = {
            "upload_id": upload_id,
            "total_files": len(files),
            "successful": successful,
            "failed": failed,
            "completed_at": datetime.now().isoformat(),
            "files": results,
        }

        logger.info(
            f"Upload {upload_id} completed: {successful} successful, {failed} failed"
        )

        await safe_log_to_websocket(
            "success",
            f"🎉 Upload batch completed: {successful} successful, {failed} failed",
            {
                "upload_id": upload_id,
                "successful": successful,
                "failed": failed,
                "total_files": len(files),
                "processing_strategy": processing_strategy,
                "total_size_mb": total_size / 1024 / 1024,
            },
        )

        return {
            "upload_id": upload_id,
            "message": f"Processed {len(results)} files: {successful} successful, {failed} failed",
            "files": results,
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "summary": {
                "total_chunks": sum(
                    r.get("chunk_count", 0) for r in results if r["status"] == "success"
                ),
                "total_characters": sum(
                    r.get("text_length", 0) for r in results if r["status"] == "success"
                ),
                "total_processing_time": sum(
                    r.get("processing_time_seconds", 0) for r in results
                ),
                "average_processing_time": (
                    sum(r.get("processing_time_seconds", 0) for r in results)
                    / len(results)
                    if results
                    else 0
                ),
            },
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {str(e)}")


@router.get("/status/{upload_id}")
async def get_upload_status(upload_id: str):
    """
    Get status of a specific upload

    Args:
        upload_id: Upload ID to check

    Returns:
        Upload status and progress
    """
    try:
        if upload_id not in upload_status_store:
            raise HTTPException(status_code=404, detail="Upload ID not found")

        return upload_status_store[upload_id]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get upload status: {str(e)}"
        )


@router.post("/url")
async def upload_url(url: str):
    """
    Upload content from URL

    Args:
        url: URL to scrape and upload

    Returns:
        Upload status
    """
    try:
        logger.info(f"Starting URL upload: {url}")

        # Import web scraper
        from app.services.web_scraper import WebScraper

        scraper = WebScraper()
        content = await scraper.scrape_url(url)

        if not content:
            raise HTTPException(
                status_code=400, detail="Failed to scrape content from URL"
            )

        # Process as text document
        filename = f"web_content_{url.replace('://', '_').replace('/', '_')}.txt"

        result = await document_service.process_uploaded_file(
            file_content=content.encode("utf-8"),
            filename=filename,
            content_type="text/plain",
        )

        # Add URL info
        result["source_url"] = url
        result["content_length"] = len(content)

        logger.info(f"URL upload completed for {url}")

        return {
            "message": "URL content processed successfully",
            "url": url,
            "result": result,
        }

    except Exception as e:
        logger.error(f"URL upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload URL: {str(e)}")


# Document endpoints removed - these are now handled by the documents.py router
# to avoid conflicts and maintain proper separation of concerns


@router.get("/history")
async def get_upload_history():
    """
    Get history of recent uploads

    Returns:
        List of recent uploads with status
    """
    try:
        # Return last 20 uploads
        recent_uploads = list(upload_status_store.values())[-20:]
        recent_uploads.sort(key=lambda x: x.get("completed_at", ""), reverse=True)

        return {"uploads": recent_uploads, "total": len(upload_status_store)}
    except Exception as e:
        logger.error(f"Upload history error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get upload history: {str(e)}"
        )


@router.get("/system/stats")
async def get_system_stats():
    """
    Get system statistics

    Returns:
        System statistics including vector database stats
    """
    try:
        from app.services.vector_service import vector_service

        # Get vector database stats
        vector_stats = vector_service.get_collection_stats()

        # Get document stats
        documents = await document_service.list_documents()
        total_size = sum(
            doc.get("metadata", {}).get("file_size", 0) for doc in documents
        )

        return {
            "vector_database": vector_stats,
            "documents": {
                "total_documents": len(documents),
                "total_size_bytes": total_size,
                "total_size_mb": (
                    round(total_size / (1024 * 1024), 2) if total_size > 0 else 0
                ),
            },
            "uploads": {
                "total_uploads": len(upload_status_store),
                "recent_uploads": len(
                    [
                        u
                        for u in upload_status_store.values()
                        if u.get("completed_at", "2000")
                        > datetime.now().replace(hour=0, minute=0, second=0).isoformat()
                    ]
                ),
            },
        }
    except Exception as e:
        logger.error(f"System stats error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get system stats: {str(e)}"
        )


# Removed /parallel endpoint - unified /upload endpoint handles all cases intelligently


@router.get("/status/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get real-time processing status for a batch"""
    try:
        if batch_id not in batch_status_store:
            raise HTTPException(status_code=404, detail="Batch not found")

        batch_status = batch_status_store[batch_id]

        # Update elapsed time
        if batch_status.get("start_time"):
            start_time = datetime.fromisoformat(batch_status["start_time"])
            elapsed = (datetime.now() - start_time).total_seconds()
            batch_status["elapsed_time"] = elapsed

            # Update remaining time estimate
            progress = batch_status.get("overall_progress", 0)
            if progress > 0 and progress < 100:
                estimated_total = elapsed / (progress / 100)
                batch_status["remaining_time"] = max(0, estimated_total - elapsed)

        return JSONResponse(status_code=200, content=batch_status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/processing/{file_id}")
async def get_file_processing_status(file_id: str):
    """Get detailed processing status for a specific file"""
    try:
        # Since we no longer use parallel processing, return a simple status
        # In practice, you might want to implement this differently
        status = {
            "file_id": file_id,
            "status": "completed",
            "message": "Enhanced document service handles all processing synchronously",
        }
        return JSONResponse(status_code=200, content=status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get file processing status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


# Removed duplicate endpoints that were causing message duplication
# The main /upload endpoint handles all cases now


@router.get("/batches")
async def list_processing_batches():
    """List all processing batches with their status"""
    try:
        batches = []
        for batch_id, status in batch_status_store.items():
            # Update elapsed time
            if status.get("start_time"):
                start_time = datetime.fromisoformat(status["start_time"])
                elapsed = (datetime.now() - start_time).total_seconds()
                status["elapsed_time"] = elapsed

            batches.append(
                {
                    "batch_id": batch_id,
                    "status": status.get("status", "unknown"),
                    "total_files": status.get("total_files", 0),
                    "completed_files": status.get("completed_files", 0),
                    "overall_progress": status.get("overall_progress", 0),
                    "elapsed_time": status.get("elapsed_time", 0),
                    "start_time": status.get("start_time"),
                }
            )

        return JSONResponse(
            status_code=200, content={"batches": batches, "total_batches": len(batches)}
        )

    except Exception as e:
        logger.error(f"Failed to list batches: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list batches: {str(e)}")


@router.delete("/batch/{batch_id}")
async def delete_batch_status(batch_id: str):
    """Delete batch status from memory"""
    try:
        if batch_id in batch_status_store:
            del batch_status_store[batch_id]
            return JSONResponse(
                status_code=200, content={"message": "Batch status deleted"}
            )
        else:
            raise HTTPException(status_code=404, detail="Batch not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete batch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete batch: {str(e)}")
