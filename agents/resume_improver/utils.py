"""
Utility functions for resume improver system
"""

import os
import logging
import base64
import requests
from typing import Optional, Tuple, List, Dict
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

# Face detection utilities
def detect_face_in_embedded_images(embedded_images: List[Dict]) -> Dict:
    """
    Detect faces in embedded images using Nebius LlaVA
    
    Args:
        embedded_images: List of embedded image dictionaries with base64 data
        
    Returns:
        Dictionary with detection results
    """
    logger = setup_logging()
    
    if not embedded_images:
        return {
            "face_found": False,
            "message": "No embedded images found in resume",
            "face_image": None
        }
    
    # Get Nebius API key
    nebius_api_key = os.getenv("NEBIUS_API_KEY")
    if not nebius_api_key:
        logger.error("NEBIUS_API_KEY not found in environment variables")
        return {
            "face_found": False,
            "message": "Face detection service not configured",
            "face_image": None
        }
    
    # Check each embedded image for faces
    for img in embedded_images:
        try:
            logger.info(f"Checking image {img.get('page_number', 'unknown')} for faces")
            
            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {nebius_api_key}",
                "Content-Type": "application/json"
            }
            
            # Create the message content for Qwen-VL
            payload = {
                "model": "Qwen/Qwen2-VL-72B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Does this image contain a human face? Answer only YES or NO."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img['base64']}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 10,
                "temperature": 0
            }
            
            # Make API request to Nebius
            response = requests.post(
                "https://api.studio.nebius.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip().upper()
                
                logger.info(f"Face detection result for image {img.get('page_number', 'unknown')}: {answer}")
                
                if "YES" in answer:
                    return {
                        "face_found": True,
                        "message": "Profile picture detected - Ready for enhancement",
                        "face_image": img
                    }
            else:
                logger.error(f"Nebius API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error checking image for faces: {e}")
            continue
    
    # No faces found in any image
    return {
        "face_found": False,
        "message": "No profile picture found in embedded images",
        "face_image": None
    }