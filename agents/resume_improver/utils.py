"""
Utility functions for resume improver system
"""

import os
import logging
from typing import Optional, Tuple, List
from pathlib import Path

# Configuration constants
MAX_FILE_SIZE_MB = 10
SUPPORTED_FORMATS = ['.pdf']
MAX_PAGES = 20

# Image processing configuration for Claude analysis
MAX_PAGE_IMAGES_FOR_ANALYSIS = 3  # Maximum page images to send to Claude
IMAGE_DPI_FOR_ANALYSIS = 150  # DPI for page images sent to Claude
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB max per image for Claude
ENABLE_VISUAL_ANALYSIS_BY_DEFAULT = True  # Whether to enable visual analysis by default

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def validate_pdf_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate uploaded PDF file
    
    Args:
        file_path: Path to the uploaded file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path:
        return False, "No file provided"
    
    # Check if file exists
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check file extension
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {file_ext}. Only PDF files are supported."
    
    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large: {file_size_mb:.1f}MB. Maximum size is {MAX_FILE_SIZE_MB}MB."
    
    return True, ""

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe processing"""
    # Remove path separators and other potentially dangerous characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
    return "".join(c for c in filename if c in safe_chars)

def create_temp_directory() -> str:
    """Create temporary directory for processing"""
    import tempfile
    return tempfile.mkdtemp(prefix="resume_improver_")

def cleanup_temp_directory(temp_dir: str):
    """Clean up temporary directory"""
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

# Error handling utilities
class ResumeProcessingError(Exception):
    """Custom exception for resume processing errors"""
    pass

class PDFValidationError(ResumeProcessingError):
    """Exception for PDF validation errors"""
    pass

class PDFProcessingError(ResumeProcessingError):
    """Exception for PDF processing errors"""
    pass