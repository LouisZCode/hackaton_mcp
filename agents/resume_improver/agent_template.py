"""
Basic LangGraph agent template for resume text analysis using Claude
"""

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from typing import Dict, List, TypedDict, Optional

# For the Graph and State
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# For observability
from langfuse.callback import CallbackHandler
import yaml
import logging
import datetime

from utils import (
    setup_logging, 
    ResumeProcessingError, 
    MAX_PAGE_IMAGES_FOR_ANALYSIS,
    IMAGE_DPI_FOR_ANALYSIS,
    MAX_IMAGE_SIZE_BYTES,
    ENABLE_VISUAL_ANALYSIS_BY_DEFAULT
)
import re

# Setup
load_dotenv()
logger = setup_logging()

# Name extraction utilities
def extract_name_from_pdf_author(pdf_data: Dict) -> str:
    """Extract person's name from PDF author metadata"""
    try:
        if not pdf_data or 'pdf_info' not in pdf_data:
            logger.warning("No PDF data available for author extraction")
            return "unknown"
            
        pdf_info = pdf_data['pdf_info']
        author = pdf_info.get('author', '')
        
        # Clean and validate author field
        if author and author.strip() and author.lower() not in ['unknown', '', 'none', 'null']:
            cleaned_author = author.strip()
            logger.info(f"Extracted name from PDF author: {cleaned_author}")
            return clean_name_for_trace(cleaned_author)
        
        logger.warning("PDF author field is empty or invalid")
        return "unknown"
        
    except Exception as e:
        logger.error(f"PDF author extraction failed: {e}")
        return "unknown"

def clean_name_for_trace(name: str) -> str:
    """Clean and format name for LangFuse trace compatibility"""
    try:
        # Convert to lowercase
        cleaned = name.lower()
        
        # Replace spaces with underscores
        cleaned = cleaned.replace(' ', '_')
        
        # Remove special characters, keep only letters, numbers, underscores
        cleaned = re.sub(r'[^a-z0-9_]', '', cleaned)
        
        # Remove multiple underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        
        # Limit length for trace compatibility
        if len(cleaned) > 30:
            cleaned = cleaned[:30]
            
        return cleaned if cleaned else "unknown"
        
    except Exception as e:
        logger.error(f"Name cleaning failed: {e}")
        return "unknown"

def create_personalized_trace_name(pdf_data: Dict) -> str:
    """Create personalized trace name using PDF author"""
    extracted_name = extract_name_from_pdf_author(pdf_data)
    
    if extracted_name and extracted_name != "unknown":
        trace_name = f"resume_{extracted_name}"
    else:
        trace_name = "resume_unknown"
    
    logger.info(f"Created trace name: {trace_name}")
    return trace_name

# LangFuse setup
def create_langfuse_handler(session_id: str, pdf_data: Dict = None, user_id: str = "resume_user"):
    """Create LangFuse callback handler with personalized trace name"""
    try:
        # Create personalized trace name using PDF author
        trace_name = create_personalized_trace_name(pdf_data) if pdf_data else "resume_unknown"
        
        handler = CallbackHandler(
            session_id=session_id, 
            user_id=user_id, 
            trace_name=trace_name
        )
        logger.info(f"LangFuse handler created - Session: {session_id}, Trace: {trace_name}")
        return handler
    except Exception as e:
        logger.warning(f"LangFuse handler creation failed: {e}")
        return None

class ResumeAnalysisState(TypedDict):
    """State structure for resume analysis workflow"""
    messages: List[AnyMessage]
    pdf_data: Optional[Dict]  # Complete PDF processing results
    resume_text: Optional[str]  # Extracted text
    analysis_results: Optional[Dict]  # Claude's analysis
    session_id: Optional[str]
    processing_status: str  # "pending", "analyzing", "complete", "error"
    error_message: Optional[str]
    visual_analysis_enabled: bool  # Whether to include page images in analysis
    image_processing_status: Optional[str]  # "pending", "processing", "complete", "error"

class ResumeAnalyzerAgent:
    """Basic LangGraph agent for resume text analysis"""
    
    def __init__(self, model_name: str = "claude-3-7-sonnet-latest"):
        self.model_name = model_name
        self.memory = MemorySaver()
        self.system_prompt = self._load_system_prompt()
        
        # Initialize Claude (callbacks will be added per request)
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=0,
            max_tokens=4000
        )
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _load_system_prompt(self) -> str:
        """Load system prompt from YAML file"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_path = os.path.join(script_dir, 'prompts.yaml')
            
            with open(yaml_path, 'r') as file:
                prompts = yaml.safe_load(file)
            
            system_prompt = prompts.get('resume_analyzer_system_prompt', '')
            
            if not system_prompt:
                logger.warning("System prompt not found in YAML, using fallback")
                return self._fallback_system_prompt()
            
            logger.info("Successfully loaded system prompt from prompts.yaml")
            return system_prompt
            
        except Exception as e:
            logger.error(f"Failed to load system prompt from YAML: {e}")
            return self._fallback_system_prompt()
    
    def _fallback_system_prompt(self) -> str:
        """Fallback system prompt if YAML loading fails"""
        return """You are an expert resume analyzer and career consultant. 

Your task is to analyze resume text and provide detailed, actionable feedback for improvement.

Analyze the resume for:
1. **Structure & Organization**: How well-organized and easy to read is it?
2. **Content Quality**: Strength of experience descriptions, achievements, skills
3. **Professional Presentation**: Language, formatting, professionalism
4. **Keywords & ATS Optimization**: Industry-relevant keywords and ATS-friendliness
5. **Overall Effectiveness**: How likely is this resume to get interviews?

Provide specific, actionable recommendations for improvement.

Format your response as structured analysis with clear sections and bullet points."""

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def analyze_resume_node(state: ResumeAnalysisState) -> ResumeAnalysisState:
            """Main analysis node - sends resume text and optionally images to Claude"""
            try:
                logger.info("Starting resume analysis")
                
                # Update status
                state["processing_status"] = "analyzing"
                state["image_processing_status"] = "processing"
                
                # Get resume text
                resume_text = state.get("resume_text", "")
                if not resume_text:
                    raise ResumeProcessingError("No resume text provided for analysis")
                
                # Check if visual analysis is enabled
                visual_analysis_enabled = state.get("visual_analysis_enabled", ENABLE_VISUAL_ANALYSIS_BY_DEFAULT)
                pdf_data = state.get("pdf_data")
                
                # Prepare message content
                if visual_analysis_enabled and pdf_data and "page_images" in pdf_data:
                    logger.info("Preparing multi-modal analysis with page images")
                    message_content = self._prepare_multimodal_content(resume_text, pdf_data)
                    analysis_type = "multimodal"
                else:
                    logger.info("Preparing text-only analysis")
                    message_content = f"""Please analyze this resume text and provide detailed improvement recommendations:

RESUME TEXT:
{resume_text}

Please provide a comprehensive analysis with specific suggestions for improvement."""
                    analysis_type = "text_only"
                
                # Create messages for Claude
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=message_content)
                ]
                
                # Get Claude's response with LangFuse tracking
                logger.info(f"Sending {analysis_type} content to Claude for analysis")
                
                # Create LLM instance (tracing handled at graph level)
                llm = ChatAnthropic(
                    model=self.model_name,
                    temperature=0,
                    max_tokens=6000,  # Increased for visual analysis
                )
                
                response = llm.invoke(messages)
                
                # Extract response content
                analysis_text = self._extract_response_content(response)
                
                # Get session ID from state
                session_id = state.get("session_id", "unknown")
                
                # Store results
                analysis_results = {
                    "analysis_text": analysis_text,
                    "model_used": self.model_name,
                    "analysis_type": analysis_type,
                    "visual_analysis_enabled": visual_analysis_enabled,
                    "input_length": len(resume_text),
                    "response_length": len(analysis_text),
                    "timestamp": str(datetime.datetime.now()),
                    "langfuse_session": session_id,
                    "trace_name": create_personalized_trace_name(pdf_data) if pdf_data else "resume_unknown"
                }
                
                # Update state
                state["messages"] = messages + [response]
                state["analysis_results"] = analysis_results
                state["processing_status"] = "complete"
                state["image_processing_status"] = "complete"
                
                logger.info(f"Resume analysis completed successfully ({analysis_type})")
                return state
                
            except Exception as e:
                logger.error(f"Resume analysis failed: {str(e)}")
                state["processing_status"] = "error"
                state["image_processing_status"] = "error"
                state["error_message"] = str(e)
                return state
        
        def check_completion(state: ResumeAnalysisState) -> str:
            """Check if analysis is complete"""
            status = state.get("processing_status", "pending")
            if status == "complete":
                return END
            elif status == "error":
                return END
            else:
                return "analyze_resume"
        
        # Build the graph
        builder = StateGraph(ResumeAnalysisState)
        
        # Add nodes
        builder.add_node("analyze_resume", analyze_resume_node)
        
        # Add edges
        builder.add_edge(START, "analyze_resume")
        builder.add_conditional_edges(
            "analyze_resume",
            check_completion
        )
        
        return builder.compile(checkpointer=self.memory)
    
    def _create_graph_with_tracing(self, session_id: str, pdf_data: Dict = None):
        """Create graph instance with LangFuse tracing"""
        # Create LangFuse handler for workflow-level tracing
        langfuse_handler = create_langfuse_handler(session_id, pdf_data)
        
        if langfuse_handler:
            # Compile graph with LangFuse callbacks
            graph_with_tracing = self.graph.with_config({
                "callbacks": [langfuse_handler],
                "metadata": {
                    "session_id": session_id,
                    "trace_name": create_personalized_trace_name(pdf_data) if pdf_data else "resume_unknown",
                    "workflow_type": "resume_analysis"
                }
            })
            logger.info(f"Graph compiled with LangFuse tracing for session: {session_id}")
            return graph_with_tracing
        else:
            logger.warning("LangFuse handler not available, using graph without tracing")
            return self.graph
    
    def _prepare_multimodal_content(self, resume_text: str, pdf_data: Dict) -> List:
        """Prepare multi-modal content combining text and images for Claude"""
        try:
            page_images = pdf_data.get("page_images", [])
            
            # Limit number of images to send
            images_to_send = page_images[:MAX_PAGE_IMAGES_FOR_ANALYSIS]
            logger.info(f"Preparing {len(images_to_send)} page images for analysis")
            
            # Start with text content
            content_parts = [
                {
                    "type": "text",
                    "text": f"""Please analyze this resume comprehensively, including both content and visual presentation:

RESUME TEXT:
{resume_text}

VISUAL ANALYSIS:
I'm also providing page images of this resume. Please analyze the visual design, layout, typography, and overall professional presentation in addition to the content analysis.

Please provide detailed recommendations for both content improvements and visual design enhancements."""
                }
            ]
            
            # Add images
            for i, img_data in enumerate(images_to_send):
                # Check image size
                base64_data = img_data.get("base64", "")
                if len(base64_data.encode()) > MAX_IMAGE_SIZE_BYTES:
                    logger.warning(f"Image {i+1} too large, skipping")
                    continue
                
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_data
                    }
                })
                
                # Add context for each image
                content_parts.append({
                    "type": "text", 
                    "text": f"Page {img_data.get('page_number', i+1)} of the resume above."
                })
            
            logger.info(f"Prepared multi-modal content with {len([p for p in content_parts if p['type'] == 'image'])} images")
            return content_parts
            
        except Exception as e:
            logger.error(f"Failed to prepare multi-modal content: {e}")
            # Fallback to text-only
            return f"""Please analyze this resume text and provide detailed improvement recommendations:

RESUME TEXT:
{resume_text}

Please provide a comprehensive analysis with specific suggestions for improvement."""
    
    def _extract_response_content(self, response) -> str:
        """Extract text content from Claude response"""
        try:
            logger.info(f"Claude response type: {type(response)}")
            
            # Safely extract content
            if hasattr(response, 'content'):
                response_content = response.content
                logger.info(f"Response content type: {type(response_content)}")
                
                # Handle different content types
                if isinstance(response_content, str):
                    analysis_text = response_content
                elif isinstance(response_content, list) and len(response_content) > 0:
                    # If content is a list, try to extract text parts
                    text_parts = []
                    for part in response_content:
                        if isinstance(part, dict) and 'text' in part:
                            text_parts.append(part['text'])
                        elif isinstance(part, str):
                            text_parts.append(part)
                    analysis_text = "\n".join(text_parts)
                else:
                    analysis_text = str(response_content)
            else:
                analysis_text = str(response)
            
            logger.info(f"Final analysis text length: {len(analysis_text)}")
            return analysis_text
            
        except Exception as e:
            logger.error(f"Failed to extract response content: {e}")
            return f"Error extracting response: {str(e)}"
    
    def analyze_resume_text(self, resume_text: str, session_id: str = None) -> Dict:
        """
        Analyze resume text using Claude
        
        Args:
            resume_text: Extracted text from PDF
            session_id: Optional session ID for tracking
            
        Returns:
            Analysis results dictionary
        """
        try:
            if not session_id:
                import uuid
                session_id = f"resume_{uuid.uuid4().hex[:8]}"
            
            # Create LangFuse handler using PDF data for author extraction
            langfuse_handler = create_langfuse_handler(session_id, pdf_data)
            logger.info(f"Created LangFuse handler for session: {session_id}")
            
            # Initial state
            initial_state = {
                "messages": [],
                "pdf_data": None,
                "resume_text": resume_text,
                "analysis_results": None,
                "session_id": session_id,
                "processing_status": "pending",
                "error_message": None,
                "visual_analysis_enabled": False,  # Text-only analysis
                "image_processing_status": None
            }
            
            # Create graph with workflow-level tracing
            graph_with_tracing = self._create_graph_with_tracing(session_id, None)
            
            # Run the analysis
            logger.info(f"Starting resume analysis for session: {session_id}")
            final_state = graph_with_tracing.invoke(
                initial_state,
                config={"configurable": {"thread_id": session_id}}
            )
            
            # Return results
            if final_state["processing_status"] == "complete":
                return {
                    "success": True,
                    "analysis": final_state["analysis_results"]["analysis_text"],
                    "metadata": {
                        "model_used": final_state["analysis_results"]["model_used"],
                        "analysis_type": final_state["analysis_results"].get("analysis_type", "text_only"),
                        "visual_analysis_enabled": final_state["analysis_results"].get("visual_analysis_enabled", False),
                        "input_length": final_state["analysis_results"]["input_length"],
                        "response_length": final_state["analysis_results"]["response_length"],
                        "session_id": session_id
                    }
                }
            else:
                return {
                    "success": False,
                    "error": final_state.get("error_message", "Unknown error"),
                    "session_id": session_id
                }
                
        except Exception as e:
            logger.error(f"Resume analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    def analyze_complete_pdf_data(self, pdf_data: Dict, session_id: str = None) -> Dict:
        """
        Analyze complete PDF data (convenience method)
        
        Args:
            pdf_data: Complete PDF processing results from pdf_processor
            session_id: Optional session ID
            
        Returns:
            Analysis results
        """
        try:
            # Extract text from PDF data
            text_data = pdf_data.get("text_data", {})
            resume_text = text_data.get("total_text", "")
            
            if not resume_text:
                return {
                    "success": False,
                    "error": "No text found in PDF data"
                }
            
            # Run analysis with PDF data for author extraction
            return self.analyze_resume_with_pdf_data(resume_text, pdf_data, session_id)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to process PDF data: {str(e)}"
            }
    
    def analyze_resume_with_pdf_data(self, resume_text: str, pdf_data: Dict, session_id: str = None) -> Dict:
        """
        Analyze resume text with PDF data for better LangFuse tracking
        
        Args:
            resume_text: Extracted text from PDF
            pdf_data: Complete PDF data for author extraction
            session_id: Optional session ID
            
        Returns:
            Analysis results dictionary
        """
        try:
            if not session_id:
                import uuid
                session_id = f"resume_{uuid.uuid4().hex[:8]}"
            
            # Initial state
            initial_state = {
                "messages": [],
                "pdf_data": pdf_data,  # Include PDF data in state
                "resume_text": resume_text,
                "analysis_results": None,
                "session_id": session_id,
                "processing_status": "pending",
                "error_message": None,
                "visual_analysis_enabled": ENABLE_VISUAL_ANALYSIS_BY_DEFAULT,  # Enable visual analysis with PDF data
                "image_processing_status": "pending"
            }
            
            # Create graph with workflow-level tracing
            graph_with_tracing = self._create_graph_with_tracing(session_id, pdf_data)
            
            # Run the analysis
            logger.info(f"Starting resume analysis for session: {session_id}")
            final_state = graph_with_tracing.invoke(
                initial_state,
                config={"configurable": {"thread_id": session_id}}
            )
            
            # Return results
            if final_state["processing_status"] == "complete":
                return {
                    "success": True,
                    "analysis": final_state["analysis_results"]["analysis_text"],
                    "metadata": {
                        "model_used": final_state["analysis_results"]["model_used"],
                        "analysis_type": final_state["analysis_results"].get("analysis_type", "text_only"),
                        "visual_analysis_enabled": final_state["analysis_results"].get("visual_analysis_enabled", False),
                        "input_length": final_state["analysis_results"]["input_length"],
                        "response_length": final_state["analysis_results"]["response_length"],
                        "session_id": session_id
                    }
                }
            else:
                return {
                    "success": False,
                    "error": final_state.get("error_message", "Unknown error"),
                    "session_id": session_id
                }
                
        except Exception as e:
            logger.error(f"Resume analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    def analyze_resume_with_prompt(self, pdf_data: Dict, prompt_key: str, session_id: str = None) -> Dict:
        """
        Analyze resume using a specific prompt from prompts.yaml
        
        Args:
            pdf_data: Complete PDF processing results
            prompt_key: Key of the prompt to use (e.g., 'resume_analysis_prompt')
            session_id: Optional session ID
            
        Returns:
            Analysis results dictionary
        """
        try:
            if not session_id:
                import uuid
                session_id = f"prompt_analysis_{uuid.uuid4().hex[:8]}"
            
            # Extract author name for personalized tracing
            author_name = extract_name_from_pdf_author(pdf_data)
            
            # Create specialized trace name based on prompt type
            if prompt_key == "resume_analysis_prompt":
                trace_name = f"resume_analysis_{author_name}"
            elif prompt_key == "resume_optimization_prompt":
                trace_name = f"resume_optimization_{author_name}"
            else:
                trace_name = f"resume_{prompt_key}_{author_name}"
            
            # Create LangFuse handler with specialized trace name
            langfuse_handler = CallbackHandler(
                session_id=session_id,
                user_id="resume_user",
                trace_name=trace_name
            )
            logger.info(f"Created specialized LangFuse handler - Session: {session_id}, Trace: {trace_name}")
            
            # Load the specific prompt
            prompt_content = self._load_prompt(prompt_key)
            if not prompt_content:
                return {
                    "success": False,
                    "error": f"Prompt '{prompt_key}' not found in prompts.yaml",
                    "session_id": session_id
                }
            
            # Prepare content for analysis
            resume_text = pdf_data["text_data"]["total_text"]
            page_images = pdf_data.get("page_images", [])
            
            # Create multi-modal content if images available
            if page_images and ENABLE_VISUAL_ANALYSIS_BY_DEFAULT:
                # Use existing multimodal content preparation
                multimodal_content = self._prepare_multimodal_content(resume_text, pdf_data)
                # Replace the hardcoded prompt with our custom prompt
                if isinstance(multimodal_content, list) and len(multimodal_content) > 0:
                    # Update the first text part with our custom prompt
                    multimodal_content[0]["text"] = f"{prompt_content}\n\nRESUME TEXT:\n{resume_text}\n\nVISUAL ANALYSIS:\nI'm also providing page images of this resume. Please analyze the visual design, layout, typography, and overall professional presentation in addition to the content analysis."
                    content = multimodal_content
                else:
                    content = f"{prompt_content}\n\nRESUME TEXT:\n{resume_text}"
                analysis_type = "multimodal"
            else:
                content = f"{prompt_content}\n\nRESUME TEXT:\n{resume_text}"
                analysis_type = "text_only"
            
            # Run analysis with LangFuse tracing
            logger.info(f"Running {analysis_type} analysis with prompt: {prompt_key}")
            
            # Create the message
            message = HumanMessage(content=content)
            
            # Invoke Claude with tracing (using callback configuration)
            response = self.llm.invoke(
                [message],
                config={"callbacks": [langfuse_handler]} if langfuse_handler else {}
            )
            
            # Extract response text
            analysis_text = self._extract_response_content(response)
            
            # Prepare results
            results = {
                "success": True,
                "analysis": analysis_text,
                "metadata": {
                    "model_used": self.model_name,
                    "analysis_type": analysis_type,
                    "prompt_key": prompt_key,
                    "visual_analysis_enabled": analysis_type == "multimodal",
                    "input_length": len(resume_text),
                    "response_length": len(analysis_text),
                    "session_id": session_id,
                    "trace_name": trace_name,
                    "author_name": author_name,
                    "timestamp": str(datetime.datetime.now())
                }
            }
            
            logger.info(f"Prompt-based analysis completed successfully for session: {session_id}")
            return results
            
        except Exception as e:
            logger.error(f"Prompt-based analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    def optimize_resume_with_prompt(self, pdf_data: Dict, previous_analysis: str, prompt_key: str, session_id: str = None) -> Dict:
        """
        Optimize resume content using optimization prompt and previous analysis
        
        Args:
            pdf_data: Complete PDF processing results
            previous_analysis: Results from the analysis step
            prompt_key: Key of the optimization prompt to use
            session_id: Optional session ID
            
        Returns:
            Optimization results dictionary
        """
        try:
            if not session_id:
                import uuid
                session_id = f"prompt_optimization_{uuid.uuid4().hex[:8]}"
            
            # Extract author name for personalized tracing
            author_name = extract_name_from_pdf_author(pdf_data)
            
            # Create specialized trace name for optimization
            trace_name = f"resume_optimization_{author_name}"
            
            # Create LangFuse handler with specialized trace name
            langfuse_handler = CallbackHandler(
                session_id=session_id,
                user_id="resume_user",
                trace_name=trace_name
            )
            logger.info(f"Created optimization LangFuse handler - Session: {session_id}, Trace: {trace_name}")
            
            # Load the optimization prompt
            prompt_content = self._load_prompt(prompt_key)
            if not prompt_content:
                return {
                    "success": False,
                    "error": f"Optimization prompt '{prompt_key}' not found in prompts.yaml",
                    "session_id": session_id
                }
            
            # Prepare content for optimization
            resume_text = pdf_data["text_data"]["total_text"]
            page_images = pdf_data.get("page_images", [])
            
            # Create comprehensive optimization prompt
            optimization_content = f"""{prompt_content}

PREVIOUS ANALYSIS RESULTS:
{previous_analysis}

ORIGINAL RESUME CONTENT:
{resume_text}

Based on the analysis above, please provide optimized resume content following the guidelines in the prompt."""
            
            # Add images if available
            if page_images and ENABLE_VISUAL_ANALYSIS_BY_DEFAULT:
                # Use existing multimodal content preparation
                multimodal_content = self._prepare_multimodal_content(resume_text, pdf_data)
                # Replace the hardcoded prompt with our optimization content
                if isinstance(multimodal_content, list) and len(multimodal_content) > 0:
                    # Update the first text part with our optimization prompt
                    multimodal_content[0]["text"] = optimization_content
                    content = multimodal_content
                else:
                    content = optimization_content
                analysis_type = "multimodal"
            else:
                content = optimization_content
                analysis_type = "text_only"
            
            # Run optimization with LangFuse tracing
            logger.info(f"Running {analysis_type} optimization with prompt: {prompt_key}")
            
            # Create the message
            message = HumanMessage(content=content)
            
            # Invoke Claude with tracing (using callback configuration)
            response = self.llm.invoke(
                [message],
                config={"callbacks": [langfuse_handler]} if langfuse_handler else {}
            )
            
            # Extract response text
            optimization_text = self._extract_response_content(response)
            
            # Prepare results
            results = {
                "success": True,
                "analysis": optimization_text,  # Using 'analysis' key for consistency
                "metadata": {
                    "model_used": self.model_name,
                    "analysis_type": analysis_type,
                    "prompt_key": prompt_key,
                    "visual_analysis_enabled": analysis_type == "multimodal",
                    "input_length": len(resume_text),
                    "response_length": len(optimization_text),
                    "session_id": session_id,
                    "trace_name": trace_name,
                    "author_name": author_name,
                    "timestamp": str(datetime.datetime.now()),
                    "based_on_analysis": True
                }
            }
            
            logger.info(f"Prompt-based optimization completed successfully for session: {session_id}")
            return results
            
        except Exception as e:
            logger.error(f"Prompt-based optimization failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    def _load_prompt(self, prompt_key: str) -> str:
        """Load a specific prompt from prompts.yaml"""
        try:
            import yaml
            from pathlib import Path
            
            # Get the directory where this script is located
            current_dir = Path(__file__).parent
            prompts_file = current_dir / "prompts.yaml"
            
            if not prompts_file.exists():
                logger.error(f"Prompts file not found: {prompts_file}")
                return None
            
            with open(prompts_file, 'r', encoding='utf-8') as f:
                prompts_data = yaml.safe_load(f)
            
            prompt_content = prompts_data.get(prompt_key)
            if not prompt_content:
                logger.error(f"Prompt key '{prompt_key}' not found in prompts.yaml")
                return None
            
            logger.info(f"Successfully loaded prompt: {prompt_key}")
            return prompt_content
            
        except Exception as e:
            logger.error(f"Failed to load prompt '{prompt_key}': {e}")
            return None

# Convenience function for simple usage
def analyze_resume(text: str, session_id: str = None) -> Dict:
    """
    Simple function to analyze resume text
    
    Args:
        text: Resume text to analyze
        session_id: Optional session ID
        
    Returns:
        Analysis results dictionary
    """
    analyzer = ResumeAnalyzerAgent()
    return analyzer.analyze_resume_text(text, session_id)

# Test function
def test_agent():
    """Test the agent with sample text"""
    sample_text = """
    John Doe
    Software Engineer
    
    Experience:
    - Worked at ABC Company for 2 years
    - Built some applications
    - Used Python and JavaScript
    
    Education:
    - Computer Science degree
    - University of Example
    """
    
    result = analyze_resume(sample_text, "test_session")
    print("Analysis Result:", result)

if __name__ == "__main__":
    test_agent()