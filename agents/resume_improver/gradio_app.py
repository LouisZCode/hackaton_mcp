"""
Gradio interface for PDF resume improvement - Testing Phase
Workflow: Upload → Deconstruct → Display Results
"""

import gradio as gr
import os
import json
from typing import Optional, Dict, List, Tuple
import logging

from pdf_processor import PDFProcessor, process_pdf
from utils import setup_logging, PDFProcessingError, PDFValidationError, format_file_size
from agent_template import ResumeAnalyzerAgent
import requests
import base64

# Setup logging
logger = setup_logging()

class ResumeImproverApp:
    """Main Gradio application for resume improvement"""
    
    def __init__(self):
        self.current_pdf_data = None
        self.processor = PDFProcessor()
        self.temp_image_dir = None
        self.resume_analyzer = ResumeAnalyzerAgent()
        self.current_analysis = None
        
    def create_status_box(self, title: str, success: bool) -> str:
        """Create HTML status box with checkmark or cross"""
        if success:
            return f"""
            <div style="border: 2px solid #10B981; background: #D1FAE5; padding: 15px; 
                        border-radius: 8px; text-align: center; margin: 5px; min-height: 80px;
                        display: flex; flex-direction: column; justify-content: center;">
                <h4 style="margin: 0 0 8px 0; color: #065F46; font-size: 14px;">{title}</h4>
                <span style="font-size: 28px; color: #10B981;">✓</span>
            </div>"""
        else:
            return f"""
            <div style="border: 2px solid #EF4444; background: #FEE2E2; padding: 15px; 
                        border-radius: 8px; text-align: center; margin: 5px; min-height: 80px;
                        display: flex; flex-direction: column; justify-content: center;">
                <h4 style="margin: 0 0 8px 0; color: #991B1B; font-size: 14px;">{title}</h4>
                <span style="font-size: 28px; color: #EF4444;">✗</span>
            </div>"""
    
    def validate_resume_content(self, text_content: str) -> bool:
        """Use AI to validate if the text content is a resume"""
        if not text_content or len(text_content.strip()) < 100:
            return False
            
        try:
            # Get Nebius API key
            nebius_api_key = os.getenv("NEBIUS_API_KEY")
            if not nebius_api_key:
                logger.warning("NEBIUS_API_KEY not found, using text heuristics for resume validation")
                # Fallback to keyword-based validation
                resume_keywords = ['experience', 'education', 'skills', 'work', 'employment', 
                                 'university', 'college', 'resume', 'cv', 'curriculum']
                text_lower = text_content.lower()
                keyword_count = sum(1 for keyword in resume_keywords if keyword in text_lower)
                return keyword_count >= 3
            
            # Use Nebius AI for resume validation
            headers = {
                "Authorization": f"Bearer {nebius_api_key}",
                "Content-Type": "application/json"
            }
            
            # Truncate text if too long for API
            sample_text = text_content[:2000] if len(text_content) > 2000 else text_content
            
            payload = {
                "model": "Qwen/Qwen2-VL-72B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Is this document a resume or CV? Answer only YES or NO.
                        
Document text:
{sample_text}"""
                    }
                ],
                "max_tokens": 10,
                "temperature": 0
            }
            
            response = requests.post(
                "https://api.studio.nebius.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip().upper()
                logger.info(f"Resume validation result: {answer}")
                return "YES" in answer
            else:
                logger.error(f"Resume validation API error: {response.status_code}")
                # Fallback to keyword validation
                resume_keywords = ['experience', 'education', 'skills', 'work', 'employment']
                text_lower = text_content.lower()
                keyword_count = sum(1 for keyword in resume_keywords if keyword in text_lower)
                return keyword_count >= 2
                
        except Exception as e:
            logger.error(f"Resume validation failed: {e}")
            # Fallback validation
            resume_keywords = ['experience', 'education', 'skills', 'work', 'employment']
            text_lower = text_content.lower()
            keyword_count = sum(1 for keyword in resume_keywords if keyword in text_lower)
            return keyword_count >= 2
        
    def process_uploaded_file(self, file) -> Tuple[str, str, str, str]:
        """
        Handle PDF upload with validation and status indicators
        
        Returns:
            Tuple of (error_message, text_status_box, images_status_box, embedded_status_box)
        """
        try:
            if file is None:
                return (
                    "",
                    self.create_status_box("Extracted Text", False),
                    self.create_status_box("PDF Images", False),
                    self.create_status_box("Embedded Images", False)
                )
            
            # Check if file is PDF
            if not file.name.lower().endswith('.pdf'):
                error_msg = "❌ **Invalid document type.** Please upload a PDF file."
                return (
                    error_msg,
                    self.create_status_box("Extracted Text", False),
                    self.create_status_box("PDF Images", False),
                    self.create_status_box("Embedded Images", False)
                )
            
            logger.info(f"Processing uploaded PDF: {file.name}")
            
            # Process the PDF
            self.current_pdf_data = process_pdf(file.name)
            
            # Extract data for validation
            text_data = self.current_pdf_data['text_data']
            page_images = self.current_pdf_data['page_images']
            embedded_images = self.current_pdf_data['embedded_images']
            
            # Validate extracted text
            text_content = text_data.get('total_text', '')
            has_text = bool(text_content and len(text_content.strip()) > 50)
            
            # Validate if content is a resume
            is_resume = False
            if has_text:
                is_resume = self.validate_resume_content(text_content)
            
            if not is_resume:
                error_msg = "❌ **This PDF doesn't appear to be a resume.** Please upload a valid resume."
                return (
                    error_msg,
                    self.create_status_box("Extracted Text", has_text),
                    self.create_status_box("PDF Images", len(page_images) > 0),
                    self.create_status_box("Embedded Images", len(embedded_images) > 0)
                )
            
            # All validations passed
            has_page_images = len(page_images) > 0
            has_embedded_images = len(embedded_images) > 0
            
            logger.info(f"Successfully processed resume: {file.name}")
            logger.info(f"Text: {has_text}, Images: {has_page_images}, Embedded: {has_embedded_images}")
            
            return (
                "",  # No error
                self.create_status_box("Extracted Text", has_text),
                self.create_status_box("PDF Images", has_page_images),
                self.create_status_box("Embedded Images", has_embedded_images)
            )
            
        except PDFValidationError as e:
            error_msg = f"❌ **PDF Validation Failed:** {str(e)}"
            logger.error(f"PDF validation failed: {str(e)}")
            return (
                error_msg,
                self.create_status_box("Extracted Text", False),
                self.create_status_box("PDF Images", False),
                self.create_status_box("Embedded Images", False)
            )
            
        except PDFProcessingError as e:
            error_msg = f"❌ **PDF Processing Failed:** {str(e)}"
            logger.error(f"PDF processing failed: {str(e)}")
            return (
                error_msg,
                self.create_status_box("Extracted Text", False),
                self.create_status_box("PDF Images", False),
                self.create_status_box("Embedded Images", False)
            )
            
        except Exception as e:
            error_msg = f"❌ **Processing Error:** {str(e)}"
            logger.error(f"Unexpected error: {str(e)}")
            return (
                error_msg,
                self.create_status_box("Extracted Text", False),
                self.create_status_box("PDF Images", False),
                self.create_status_box("Embedded Images", False)
            )
    
    def _save_temp_image(self, base64_data: str, filename: str) -> str:
        """Save base64 image to temp file with short path"""
        import tempfile
        import base64
        
        # Create temp directory if needed - use system temp for Gradio security
        if not self.temp_image_dir:
            # Always use system temp directory for Gradio compatibility
            self.temp_image_dir = tempfile.mkdtemp(prefix="resume_", suffix="_imgs")
        
        # Decode base64 image
        image_bytes = base64.b64decode(base64_data)
        
        # Save to temp file with short name
        temp_path = os.path.join(self.temp_image_dir, filename)
        
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        return temp_path
    
    
    def get_image_analysis(self) -> str:
        """Get detailed image analysis with face detection for the current PDF using LangGraph workflow"""
        if not self.current_pdf_data:
            return "No PDF processed yet."
        
        embedded_images = self.current_pdf_data['embedded_images']
        
        if not embedded_images:
            return "## 🖼️ **Image Analysis**\n\n❌ **No embedded images found in resume**"
        
        # Import face detection agent
        from utils import FaceDetectionAgent
        
        # Create session ID for face detection workflow
        import uuid
        session_id = f"face_detection_{uuid.uuid4().hex[:8]}"
        
        # Initialize face detection agent
        face_agent = FaceDetectionAgent()
        
        # Run face detection workflow with LangFuse tracing
        logger.info("Running face detection workflow with LangFuse tracing")
        face_detection_result = face_agent.detect_faces_with_workflow(
            embedded_images=embedded_images,
            pdf_data=self.current_pdf_data,
            session_id=session_id
        )
        
        # Build analysis markdown
        analysis_md = "## 🖼️ **Image Analysis - Face Detection**\n\n"
        
        if face_detection_result["face_found"]:
            analysis_md += f"### ✅ **{face_detection_result['message']}**\n\n"
            
            face_img = face_detection_result["face_image"]
            analysis_md += f"""
**Detected Profile Picture Details:**
- **Image:** {face_img['image_index'] + 1} (Page {face_img['page_number']})
- **Dimensions:** {face_img['width']} × {face_img['height']} pixels
- **Format:** {face_img['format']}
- **Size:** {format_file_size(face_img['size_bytes'])}
- **Aspect Ratio:** {face_img['aspect_ratio']:.2f}

*This image can be enhanced for professional presentation.*

**🔍 LangFuse Trace:** `profile_{session_id}` - Full workflow tracked for observability

"""
        else:
            analysis_md += f"### ❌ **{face_detection_result['message']}**\n\n"
            analysis_md += f"**🔍 LangFuse Trace:** `profile_{session_id}` - Workflow completed with no face detected\n\n"
        
        # Show summary of all embedded images
        analysis_md += f"**Total Embedded Images Found**: {len(embedded_images)}\n\n"
        
        for img in embedded_images:
            analysis_md += f"""
**Image {img['image_index'] + 1}** (Page {img['page_number']}):
- **Dimensions:** {img['width']} × {img['height']} pixels
- **Format:** {img['format']} 
- **Size:** {format_file_size(img['size_bytes'])}
- **Aspect Ratio:** {img['aspect_ratio']:.2f}

"""
        
        return analysis_md
    
    def analyze_resume_with_claude(self) -> str:
        """Run Claude analysis on the current resume"""
        if not self.current_pdf_data:
            return "❌ **No PDF processed yet.** Please upload and process a PDF first."
        
        try:
            logger.info("Starting Claude resume analysis")
            
            # Generate session ID for this analysis with timestamp
            import uuid
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"gradio_{timestamp}_{uuid.uuid4().hex[:6]}"
            
            logger.info(f"Created analysis session: {session_id}")
            
            # Run the analysis
            result = self.resume_analyzer.analyze_complete_pdf_data(
                self.current_pdf_data, 
                session_id=session_id
            )
            
            if result["success"]:
                self.current_analysis = result
                
                # Format the analysis for display
                metadata = result['metadata']
                analysis_type = metadata.get('analysis_type', 'text_only')
                visual_analysis_enabled = metadata.get('visual_analysis_enabled', False)
                
                # Analysis type indicator
                analysis_type_icon = "🎨 **Multi-Modal Analysis**" if analysis_type == "multimodal" else "📝 **Text-Only Analysis**"
                
                analysis_md = f"""
# 🎯 **Claude Resume Analysis**

## 📊 **Analysis Metadata**
- **Type:** {analysis_type_icon}
- **Model:** {metadata['model_used']}
- **Input Length:** {metadata['input_length']:,} characters
- **Analysis Length:** {metadata['response_length']:,} characters  
- **Visual Analysis:** {'✅ Enabled' if visual_analysis_enabled else '❌ Disabled'}
- **Session ID:** {metadata['session_id']}

---

## 📝 **Detailed Analysis & Recommendations**

{result['analysis']}

---

*💡 Analysis completed successfully! Use this feedback to improve your resume.*
"""
                
                logger.info(f"Claude analysis completed successfully for session: {session_id}")
                return analysis_md
                
            else:
                error_msg = f"""
# ❌ **Analysis Failed**

**Error:** {result['error']}

**Session ID:** {result.get('session_id', 'unknown')}

Please try again or check your API configuration.
"""
                logger.error(f"Claude analysis failed: {result['error']}")
                return error_msg
                
        except Exception as e:
            error_msg = f"""
# ❌ **Analysis Error**

**Unexpected Error:** {str(e)}

Please check your internet connection and API configuration.
"""
            logger.error(f"Claude analysis error: {str(e)}")
            return error_msg
    
    def get_analysis_summary(self) -> str:
        """Get a summary of the current analysis"""
        if not self.current_analysis:
            return "No analysis available. Run Claude analysis first."
        
        metadata = self.current_analysis["metadata"]
        
        summary = f"""
## 📋 **Analysis Summary**

- **Status:** ✅ Complete
- **Model Used:** {metadata['model_used']}
- **Resume Length:** {metadata['input_length']:,} characters
- **Analysis Length:** {metadata['response_length']:,} characters
- **Session:** {metadata['session_id']}

### 🎯 **Key Points:**
*Run full analysis above to see detailed recommendations*
"""
        return summary
    
    def export_processing_data(self) -> str:
        """Export current processing data as JSON"""
        if not self.current_pdf_data:
            return None
        
        # Create a simplified version for export (without base64 images)
        export_data = {
            'pdf_info': self.current_pdf_data['pdf_info'],
            'text_summary': {
                'total_words': self.current_pdf_data['text_data']['total_words'],
                'total_chars': self.current_pdf_data['text_data']['total_chars'],
                'pages': [
                    {
                        'page_number': page['page_number'],
                        'word_count': page['word_count'],
                        'char_count': page['char_count'],
                        'has_text': page['has_text']
                    }
                    for page in self.current_pdf_data['text_data']['pages']
                ]
            },
            'image_summary': [
                {
                    'page_number': img['page_number'],
                    'width': img['width'],
                    'height': img['height'],
                    'format': img['format'],
                    'size_bytes': img['size_bytes'],
                    'is_potential_headshot': img['is_potential_headshot']
                }
                for img in self.current_pdf_data['embedded_images']
            ],
            'processing_summary': self.current_pdf_data['processing_summary']
        }
        
        # Add Claude analysis if available
        if self.current_analysis:
            export_data['claude_analysis'] = {
                'analysis_text': self.current_analysis['analysis'],
                'metadata': self.current_analysis['metadata']
            }
        
        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f, indent=2)
            return f.name

def create_interface():
    """Create the simplified Gradio interface"""
    
    app = ResumeImproverApp()
    
    with gr.Blocks(title="Resume Improver - PDF AI processing", theme=gr.themes.Soft()) as interface:
        
        # Simple, clean header
        gr.Markdown("# Resume Improver - PDF AI processing")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Direct upload without steps
                pdf_file = gr.File(
                    label="Upload PDF Resume",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                # Error/status message area
                error_display = gr.Markdown(value="", visible=True)
            
            with gr.Column(scale=2):
                # Status indicator boxes
                with gr.Row():
                    text_status = gr.HTML(
                        value=app.create_status_box("Extracted Text", False),
                        label="Text Status"
                    )
                    images_status = gr.HTML(
                        value=app.create_status_box("PDF Images", False),
                        label="Images Status"
                    )
                    embedded_status = gr.HTML(
                        value=app.create_status_box("Embedded Images", False),
                        label="Embedded Status"
                    )
        
        # Analysis Tabs (only show when processing is successful)
        with gr.Tabs(visible=False) as analysis_tabs:
            with gr.TabItem("🎯 Experience Analysis"):
                claude_analysis_btn = gr.Button(
                    "🚀 Analyze Resume with Claude", 
                    variant="primary",
                    size="lg"
                )
                claude_analysis_display = gr.Markdown(
                    "Ready for AI analysis...",
                    label="Experience Analysis Results"
                )
            
            with gr.TabItem("🖼️ Image Analysis"):
                image_analysis_btn = gr.Button("🔍 Get Face Detection Analysis")
                image_analysis_display = gr.Markdown()
            
            with gr.TabItem("📥 Export Data"):
                export_btn = gr.Button("📥 Export Processing Data (JSON)")
                export_file = gr.File(label="Download Processing Data")
        
        def handle_file_upload(file):
            """Handle file upload with automatic processing"""
            error_msg, text_box, images_box, embedded_box = app.process_uploaded_file(file)
            
            # Show analysis tabs only if processing was successful (no error)
            show_tabs = (error_msg == "")
            
            return [
                error_msg,  # error_display
                text_box,   # text_status
                images_box, # images_status
                embedded_box, # embedded_status
                gr.update(visible=show_tabs)  # analysis_tabs
            ]
        
        # Auto-process on file upload
        pdf_file.change(
            fn=handle_file_upload,
            inputs=[pdf_file],
            outputs=[
                error_display,
                text_status,
                images_status,
                embedded_status,
                analysis_tabs
            ]
        )
        
        # Analysis event handlers
        claude_analysis_btn.click(
            fn=app.analyze_resume_with_claude,
            outputs=[claude_analysis_display]
        )
        
        image_analysis_btn.click(
            fn=app.get_image_analysis,
            outputs=[image_analysis_display]
        )
        
        export_btn.click(
            fn=app.export_processing_data,
            outputs=[export_file]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        debug=True             # Enable debug mode
    )