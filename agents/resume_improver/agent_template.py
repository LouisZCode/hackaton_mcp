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

from utils import setup_logging, ResumeProcessingError
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
        """Load system prompt - for now use a basic one"""
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
            """Main analysis node - sends resume text to Claude"""
            try:
                logger.info("Starting resume text analysis")
                
                # Update status
                state["processing_status"] = "analyzing"
                
                # Get resume text
                resume_text = state.get("resume_text", "")
                if not resume_text:
                    raise ResumeProcessingError("No resume text provided for analysis")
                
                # Create messages for Claude
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=f"""Please analyze this resume text and provide detailed improvement recommendations:

RESUME TEXT:
{resume_text}

Please provide a comprehensive analysis with specific suggestions for improvement.""")
                ]
                
                # Get Claude's response with LangFuse tracking
                logger.info(f"Sending {len(resume_text)} characters to Claude for analysis")
                
                # Create LangFuse handler for this specific request using PDF data for author
                session_id = state.get("session_id", "unknown")
                pdf_data = state.get("pdf_data")
                langfuse_handler = create_langfuse_handler(session_id, pdf_data)
                
                # Create LLM with callbacks for this request
                llm_with_callbacks = ChatAnthropic(
                    model=self.model_name,
                    temperature=0,
                    max_tokens=4000,
                    callbacks=[langfuse_handler] if langfuse_handler else []
                )
                
                response = llm_with_callbacks.invoke(messages)
                
                # Debug response object
                logger.info(f"Claude response type: {type(response)}")
                logger.info(f"Claude response attributes: {dir(response)}")
                
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
                
                # Store results
                analysis_results = {
                    "analysis_text": analysis_text,
                    "model_used": self.model_name,
                    "input_length": len(resume_text),
                    "response_length": len(analysis_text),
                    "timestamp": str(datetime.datetime.now()),
                    "langfuse_session": session_id
                }
                
                # Update state
                state["messages"] = messages + [response]
                state["analysis_results"] = analysis_results
                state["processing_status"] = "complete"
                
                logger.info("Resume analysis completed successfully")
                return state
                
            except Exception as e:
                logger.error(f"Resume analysis failed: {str(e)}")
                state["processing_status"] = "error"
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
                "error_message": None
            }
            
            # Run the analysis
            logger.info(f"Starting resume analysis for session: {session_id}")
            final_state = self.graph.invoke(
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
            
            # Create LangFuse handler using PDF data for author extraction
            langfuse_handler = create_langfuse_handler(session_id, pdf_data)
            logger.info(f"Created LangFuse handler for session: {session_id}")
            
            # Initial state
            initial_state = {
                "messages": [],
                "pdf_data": pdf_data,  # Include PDF data in state
                "resume_text": resume_text,
                "analysis_results": None,
                "session_id": session_id,
                "processing_status": "pending",
                "error_message": None
            }
            
            # Run the analysis
            logger.info(f"Starting resume analysis for session: {session_id}")
            final_state = self.graph.invoke(
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