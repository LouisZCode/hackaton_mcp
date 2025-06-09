"""
Utility functions for resume improver system
"""

import os
import logging
import base64
import requests
from typing import Optional, Tuple, List, Dict
from pathlib import Path

# LangGraph imports
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage as AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langfuse.callback import CallbackHandler

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

# Face Detection LangGraph Implementation
class FaceDetectionState(TypedDict):
    """State structure for face detection workflow"""
    embedded_images: List[Dict]
    detection_result: Optional[Dict]
    session_id: str
    pdf_data: Optional[Dict]  # For author extraction
    processing_status: str  # "pending", "processing", "complete", "error"
    error_message: Optional[str]

def create_face_detection_langfuse_handler(session_id: str, pdf_data: Dict = None, user_id: str = "face_detection_user"):
    """Create LangFuse callback handler for face detection workflow"""
    try:
        # Import the existing functions from agent_template
        from agent_template import extract_name_from_pdf_author, clean_name_for_trace
        
        # Create personalized trace name for face detection
        if pdf_data:
            extracted_name = extract_name_from_pdf_author(pdf_data)
            if extracted_name and extracted_name != "unknown":
                trace_name = f"profile_{clean_name_for_trace(extracted_name)}"
            else:
                trace_name = "profile_unknown"
        else:
            trace_name = "profile_unknown"
        
        handler = CallbackHandler(
            session_id=session_id,
            user_id=user_id,
            trace_name=trace_name
        )
        
        logger = setup_logging()
        logger.info(f"Face detection LangFuse handler created - Session: {session_id}, Trace: {trace_name}")
        return handler
        
    except Exception as e:
        logger = setup_logging()
        logger.warning(f"Face detection LangFuse handler creation failed: {e}")
        return None

class FaceDetectionAgent:
    """LangGraph agent for face detection in embedded images"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.memory = MemorySaver()
        self._current_callbacks = None  # Store current callbacks for nodes to access
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the face detection LangGraph workflow"""
        
        def detect_faces_node(state: FaceDetectionState) -> FaceDetectionState:
            """Main face detection node using Nebius API"""
            try:
                self.logger.info("Starting face detection workflow")
                state["processing_status"] = "processing"
                
                embedded_images = state.get("embedded_images", [])
                if not embedded_images:
                    state["detection_result"] = {
                        "face_found": False,
                        "message": "No embedded images found in resume",
                        "face_image": None
                    }
                    state["processing_status"] = "complete"
                    return state
                
                # Get Nebius API key
                nebius_api_key = os.getenv("NEBIUS_API_KEY")
                if not nebius_api_key:
                    self.logger.error("NEBIUS_API_KEY not found in environment variables")
                    state["detection_result"] = {
                        "face_found": False,
                        "message": "Face detection service not configured",
                        "face_image": None
                    }
                    state["processing_status"] = "error"
                    state["error_message"] = "Face detection service not configured"
                    return state
                
                # Import required modules for API calls
                import time
                
                # Get the callback handler from the agent instance
                callback_handler = getattr(self, '_current_callbacks', None)
                if callback_handler and isinstance(callback_handler, list):
                    # Find the LangFuse callback handler
                    langfuse_handler = None
                    for cb in callback_handler:
                        if isinstance(cb, CallbackHandler):
                            langfuse_handler = cb
                            break
                    callback_handler = langfuse_handler
                
                self.logger.info(f"LangFuse callback handler found: {callback_handler is not None}")
                
                # Check each embedded image for faces
                for img_index, img in enumerate(embedded_images):
                    try:
                        image_info = f"Image {img_index + 1} (Page {img.get('page_number', 'unknown')})"
                        self.logger.info(f"Checking {image_info} for faces")
                        
                        # Create child span within LangGraph context
                        span = None
                        if callback_handler:
                            try:
                                # Create a child span using the callback handler's LangFuse client
                                span = callback_handler.langfuse.span(
                                    name=f"nebius_face_detection_image_{img_index + 1}",
                                    metadata={
                                        "image_index": img_index + 1,
                                        "page_number": img.get('page_number', 'unknown'),
                                        "image_dimensions": f"{img.get('width', 0)}x{img.get('height', 0)}",
                                        "image_format": img.get('format', 'unknown'),
                                        "image_size_bytes": img.get('size_bytes', 0),
                                        "model": "Qwen/Qwen2-VL-72B-Instruct",
                                        "node_type": "face_detection_api_call"
                                    }
                                )
                                self.logger.info(f"Created child span for {image_info} within LangGraph context")
                            except Exception as span_error:
                                self.logger.warning(f"Could not create child span: {span_error}")
                                span = None
                        else:
                            # Fallback: Use direct LangFuse if no callback handler
                            try:
                                from langfuse import Langfuse
                                langfuse = Langfuse()
                                span = langfuse.span(
                                    name=f"nebius_face_detection_image_{img_index + 1}_fallback",
                                    metadata={
                                        "image_index": img_index + 1,
                                        "page_number": img.get('page_number', 'unknown'),
                                        "image_dimensions": f"{img.get('width', 0)}x{img.get('height', 0)}",
                                        "image_format": img.get('format', 'unknown'),
                                        "image_size_bytes": img.get('size_bytes', 0),
                                        "model": "Qwen/Qwen2-VL-72B-Instruct",
                                        "node_type": "face_detection_api_call_fallback"
                                    }
                                )
                                self.logger.info(f"Created fallback span for {image_info}")
                            except Exception as span_error:
                                self.logger.warning(f"Could not create fallback span: {span_error}")
                                span = None
                        
                        start_time = time.time()
                        
                        # Prepare the API request
                        headers = {
                            "Authorization": f"Bearer {nebius_api_key}",
                            "Content-Type": "application/json"
                        }
                        
                        # Create the message content for Qwen-VL (without base64 for logging)
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
                        
                        # Log request without the large image data
                        span.event(
                            name="nebius_api_request",
                            metadata={
                                "prompt": "Does this image contain a human face? Answer only YES or NO.",
                                "model": "Qwen/Qwen2-VL-72B-Instruct",
                                "max_tokens": 10,
                                "temperature": 0,
                                "image_info": image_info
                            }
                        )
                        
                        # Make API request to Nebius
                        response = requests.post(
                            "https://api.studio.nebius.com/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=30
                        )
                        
                        processing_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            result = response.json()
                            answer = result["choices"][0]["message"]["content"].strip().upper()
                            
                            # Log successful response
                            span.event(
                                name="nebius_api_response",
                                metadata={
                                    "status_code": 200,
                                    "face_detection_result": answer,
                                    "processing_time_seconds": round(processing_time, 3),
                                    "tokens_used": result.get("usage", {}).get("total_tokens", "unknown")
                                }
                            )
                            
                            self.logger.info(f"Face detection result for {image_info}: {answer}")
                            
                            if "YES" in answer:
                                # Log face found event
                                span.event(
                                    name="face_detected",
                                    metadata={
                                        "result": "FACE_FOUND",
                                        "image_selected": image_info,
                                        "confidence": "high"
                                    }
                                )
                                if span:
                                    span.end()
                                
                                state["detection_result"] = {
                                    "face_found": True,
                                    "message": "Profile picture detected - Ready for enhancement",
                                    "face_image": img
                                }
                                state["processing_status"] = "complete"
                                return state
                            else:
                                # Log no face found
                                span.event(
                                    name="no_face_detected",
                                    metadata={
                                        "result": "NO_FACE",
                                        "nebius_response": answer
                                    }
                                )
                        else:
                            # Log API error
                            span.event(
                                name="nebius_api_error",
                                metadata={
                                    "status_code": response.status_code,
                                    "error_message": response.text,
                                    "processing_time_seconds": round(processing_time, 3)
                                }
                            )
                            self.logger.error(f"Nebius API error: {response.status_code} - {response.text}")
                        
                        # End span if it exists
                        if span:
                            span.end()
                            
                    except Exception as e:
                        self.logger.error(f"Error checking image for faces: {e}")
                        continue
                
                # No faces found in any image
                state["detection_result"] = {
                    "face_found": False,
                    "message": "No profile picture found in embedded images",
                    "face_image": None
                }
                state["processing_status"] = "complete"
                return state
                
            except Exception as e:
                self.logger.error(f"Face detection workflow failed: {str(e)}")
                state["processing_status"] = "error"
                state["error_message"] = str(e)
                state["detection_result"] = {
                    "face_found": False,
                    "message": "Face detection failed",
                    "face_image": None
                }
                return state
        
        def check_completion(state: FaceDetectionState) -> str:
            """Check if face detection is complete"""
            status = state.get("processing_status", "pending")
            if status in ["complete", "error"]:
                return END
            else:
                return "detect_faces"
        
        # Build the graph
        builder = StateGraph(FaceDetectionState)
        
        # Add nodes
        builder.add_node("detect_faces", detect_faces_node)
        
        # Add edges
        builder.add_edge(START, "detect_faces")
        builder.add_conditional_edges(
            "detect_faces",
            check_completion
        )
        
        return builder.compile(checkpointer=self.memory)
    
    def detect_faces_with_workflow(self, embedded_images: List[Dict], pdf_data: Dict = None, session_id: str = None) -> Dict:
        """
        Run face detection workflow with LangFuse tracing
        
        Args:
            embedded_images: List of embedded image dictionaries
            pdf_data: PDF data for author extraction
            session_id: Optional session ID
            
        Returns:
            Face detection results dictionary
        """
        try:
            if not session_id:
                import uuid
                session_id = f"face_detection_{uuid.uuid4().hex[:8]}"
            
            # Create LangFuse handler for face detection workflow
            langfuse_handler = create_face_detection_langfuse_handler(session_id, pdf_data)
            
            # Create graph with tracing and store callbacks for nodes to access
            if langfuse_handler:
                # Store callback handler for nodes to access
                self._current_callbacks = [langfuse_handler]
                
                graph_with_tracing = self.graph.with_config({
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "session_id": session_id,
                        "workflow_type": "face_detection"
                    }
                })
                self.logger.info(f"Face detection graph compiled with LangFuse tracing for session: {session_id}")
            else:
                self._current_callbacks = None
                graph_with_tracing = self.graph
                self.logger.warning("Face detection graph running without LangFuse tracing")
            
            # Initial state
            initial_state = {
                "embedded_images": embedded_images,
                "detection_result": None,
                "session_id": session_id,
                "pdf_data": pdf_data,
                "processing_status": "pending",
                "error_message": None
            }
            
            # Run the face detection workflow
            self.logger.info(f"Starting face detection workflow for session: {session_id} with enhanced LangFuse tracing")
            final_state = graph_with_tracing.invoke(
                initial_state,
                config={"configurable": {"thread_id": session_id}}
            )
            
            # Clean up callback reference
            self._current_callbacks = None
            
            self.logger.info(f"Face detection workflow completed for session: {session_id} - Enhanced tracing complete")
            return final_state["detection_result"]
            
        except Exception as e:
            self.logger.error(f"Face detection workflow failed: {str(e)}")
            return {
                "face_found": False,
                "message": "Face detection workflow error",
                "face_image": None,
                "error": str(e)
            }