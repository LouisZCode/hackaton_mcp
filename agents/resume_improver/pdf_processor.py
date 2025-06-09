"""
PDF processing module using PyMuPDF for all PDF operations
- Text extraction
- PDF to image conversion  
- Embedded image extraction
- Metadata extraction
"""

import pymupdf  # PyMuPDF
import os
import base64
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

from utils import (
    validate_pdf_file, 
    PDFProcessingError, 
    PDFValidationError,
    format_file_size,
    MAX_PAGES
)

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Main PDF processing class using PyMuPDF"""
    
    def __init__(self):
        self.document = None
        self.file_path = None
        
    def load_pdf(self, file_path: str) -> Dict:
        """
        Load and validate PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dict with basic PDF information
        """
        try:
            # Validate file
            is_valid, error_msg = validate_pdf_file(file_path)
            if not is_valid:
                raise PDFValidationError(error_msg)
            
            # Open PDF with PyMuPDF
            self.document = pymupdf.open(file_path)
            self.file_path = file_path
            
            # Get basic info
            info = self._get_pdf_info()
            logger.info(f"Successfully loaded PDF: {info['filename']}")
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to load PDF: {str(e)}")
            raise PDFProcessingError(f"Failed to load PDF: {str(e)}")
    
    def _get_pdf_info(self) -> Dict:
        """Extract basic PDF information"""
        if not self.document:
            raise PDFProcessingError("No PDF loaded")
            
        file_stat = os.stat(self.file_path)
        metadata = self.document.metadata
        
        return {
            'filename': Path(self.file_path).name,
            'file_size': file_stat.st_size,
            'file_size_formatted': format_file_size(file_stat.st_size),
            'page_count': len(self.document),
            'title': metadata.get('title', 'Unknown'),
            'author': metadata.get('author', 'Unknown'),
            'subject': metadata.get('subject', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'creation_date': metadata.get('creationDate', ''),
            'modification_date': metadata.get('modDate', ''),
            'encrypted': self.document.needs_pass,
            'pdf_version': self._get_pdf_version()
        }
    
    def _get_pdf_version(self) -> str:
        """Safely get PDF version with fallback"""
        try:
            # Try different methods to get PDF version
            if hasattr(self.document, 'pdf_version'):
                return f"PDF {self.document.pdf_version()}"
            elif hasattr(self.document, 'metadata') and 'format' in self.document.metadata:
                return self.document.metadata.get('format', 'PDF (Unknown version)')
            else:
                return "PDF (Version unknown)"
        except Exception:
            return "PDF (Version unknown)"
    
    def extract_text(self) -> Dict:
        """
        Extract all text from PDF
        
        Returns:
            Dict with text content per page
        """
        if not self.document:
            raise PDFProcessingError("No PDF loaded")
            
        try:
            text_data = {
                'pages': [],
                'total_text': '',
                'total_chars': 0,
                'total_words': 0
            }
            
            for page_num in range(len(self.document)):
                page = self.document[page_num]
                
                # Extract plain text
                page_text = page.get_text()
                
                # Extract structured text (with position info)
                text_dict = page.get_text("dict")
                
                page_data = {
                    'page_number': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text),
                    'word_count': len(page_text.split()),
                    'blocks': len(text_dict.get('blocks', [])),
                    'has_text': bool(page_text.strip())
                }
                
                text_data['pages'].append(page_data)
                text_data['total_text'] += page_text + '\n\n'
            
            # Calculate totals
            text_data['total_chars'] = len(text_data['total_text'])
            text_data['total_words'] = len(text_data['total_text'].split())
            
            logger.info(f"Extracted text: {text_data['total_words']} words across {len(text_data['pages'])} pages")
            return text_data
            
        except Exception as e:
            logger.error(f"Failed to extract text: {str(e)}")
            raise PDFProcessingError(f"Failed to extract text: {str(e)}")
    
    def convert_to_images(self, dpi: int = 150) -> List[Dict]:
        """
        Convert PDF pages to images
        
        Args:
            dpi: Resolution for image conversion
            
        Returns:
            List of image data dictionaries
        """
        if not self.document:
            raise PDFProcessingError("No PDF loaded")
            
        try:
            images = []
            
            for page_num in range(len(self.document)):
                page = self.document[page_num]
                
                # Create pixmap (image) from page
                pix = page.get_pixmap(dpi=dpi)
                
                # Convert to bytes
                img_bytes = pix.tobytes("png")
                
                # Encode to base64 for web display
                img_base64 = base64.b64encode(img_bytes).decode()
                
                image_data = {
                    'page_number': page_num + 1,
                    'width': pix.width,
                    'height': pix.height,
                    'dpi': dpi,
                    'format': 'PNG',
                    'size_bytes': len(img_bytes),
                    'base64': img_base64,
                    'data_url': f"data:image/png;base64,{img_base64}"
                }
                
                images.append(image_data)
                
                # Clean up pixmap
                pix = None
            
            logger.info(f"Converted {len(images)} pages to images at {dpi} DPI")
            return images
            
        except Exception as e:
            logger.error(f"Failed to convert pages to images: {str(e)}")
            raise PDFProcessingError(f"Failed to convert pages to images: {str(e)}")
    
    def extract_embedded_images(self) -> List[Dict]:
        """
        Extract embedded images from PDF (like headshots)
        
        Returns:
            List of embedded image data
        """
        if not self.document:
            raise PDFProcessingError("No PDF loaded")
            
        try:
            embedded_images = []
            
            for page_num in range(len(self.document)):
                page = self.document[page_num]
                
                # Get all images on this page
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    # Extract image data
                    xref = img[0]  # Image reference number
                    
                    try:
                        # Extract the image
                        image_data = self.document.extract_image(xref)
                        
                        # Get image bytes and metadata
                        img_bytes = image_data["image"]
                        img_ext = image_data["ext"]
                        img_width = image_data.get("width", 0)
                        img_height = image_data.get("height", 0)
                        
                        # Encode to base64
                        img_base64 = base64.b64encode(img_bytes).decode()
                        
                        # Determine if this might be a headshot/portrait
                        is_potential_headshot = self._is_potential_headshot(img_width, img_height, len(img_bytes))
                        
                        embedded_img = {
                            'page_number': page_num + 1,
                            'image_index': img_index,
                            'xref': xref,
                            'width': img_width,
                            'height': img_height,
                            'format': img_ext.upper(),
                            'size_bytes': len(img_bytes),
                            'base64': img_base64,
                            'data_url': f"data:image/{img_ext};base64,{img_base64}",
                            'is_potential_headshot': is_potential_headshot,
                            'aspect_ratio': img_width / img_height if img_height > 0 else 0
                        }
                        
                        embedded_images.append(embedded_img)
                        
                    except Exception as img_error:
                        logger.warning(f"Failed to extract image {xref} on page {page_num + 1}: {str(img_error)}")
                        continue
            
            logger.info(f"Extracted {len(embedded_images)} embedded images")
            return embedded_images
            
        except Exception as e:
            logger.error(f"Failed to extract embedded images: {str(e)}")
            raise PDFProcessingError(f"Failed to extract embedded images: {str(e)}")
    
    def _is_potential_headshot(self, width: int, height: int, size_bytes: int) -> bool:
        """
        Heuristic to determine if an image might be a headshot/portrait
        
        Args:
            width: Image width in pixels
            height: Image height in pixels  
            size_bytes: Image size in bytes
            
        Returns:
            True if image characteristics suggest it could be a headshot
        """
        # Basic size filters
        if width < 50 or height < 50:  # Too small
            return False
            
        if size_bytes < 5000:  # Too small file size (< 5KB)
            return False
            
        # Aspect ratio check (portrait-ish or square)
        aspect_ratio = width / height if height > 0 else 0
        
        # Typical headshot ratios: 0.6 to 1.5 (portrait to slightly landscape)
        if 0.6 <= aspect_ratio <= 1.5:
            # Size check - reasonable dimensions for a headshot
            if (100 <= width <= 2000) and (100 <= height <= 2000):
                return True
                
        return False
    
    def process_complete_pdf(self, file_path: str, image_dpi: int = 150) -> Dict:
        """
        Complete PDF processing pipeline
        
        Args:
            file_path: Path to PDF file
            image_dpi: Resolution for page images
            
        Returns:
            Complete processing results
        """
        try:
            # Load PDF
            pdf_info = self.load_pdf(file_path)
            
            # Check page count limit
            if pdf_info['page_count'] > MAX_PAGES:
                raise PDFProcessingError(f"PDF has {pdf_info['page_count']} pages. Maximum allowed is {MAX_PAGES}.")
            
            # Extract text
            text_data = self.extract_text()
            
            # Convert pages to images
            page_images = self.convert_to_images(dpi=image_dpi)
            
            # Extract embedded images
            embedded_images = self.extract_embedded_images()
            
            # Compile results
            results = {
                'pdf_info': pdf_info,
                'text_data': text_data,
                'page_images': page_images,
                'embedded_images': embedded_images,
                'processing_summary': {
                    'total_pages': len(page_images),
                    'total_text_length': text_data['total_chars'],
                    'total_words': text_data['total_words'],
                    'embedded_images_count': len(embedded_images),
                    'potential_headshots': sum(1 for img in embedded_images if img['is_potential_headshot']),
                    'processing_success': True
                }
            }
            
            logger.info(f"Complete PDF processing finished successfully")
            return results
            
        except Exception as e:
            logger.error(f"Complete PDF processing failed: {str(e)}")
            raise PDFProcessingError(f"PDF processing failed: {str(e)}")
    
    def close(self):
        """Close the PDF document"""
        if self.document:
            self.document.close()
            self.document = None
            self.file_path = None

# Convenience function for simple processing
def process_pdf(file_path: str, image_dpi: int = 150) -> Dict:
    """
    Process a PDF file with a single function call
    
    Args:
        file_path: Path to PDF file
        image_dpi: Resolution for page images
        
    Returns:
        Complete processing results
    """
    processor = PDFProcessor()
    try:
        return processor.process_complete_pdf(file_path, image_dpi)
    finally:
        processor.close()