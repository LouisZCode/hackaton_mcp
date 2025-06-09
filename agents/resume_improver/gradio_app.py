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

# Setup logging
logger = setup_logging()

class ResumeImproverApp:
    """Main Gradio application for resume improvement"""
    
    def __init__(self):
        self.current_pdf_data = None
        self.processor = PDFProcessor()
        self.temp_image_dir = None
        
    def upload_and_process_pdf(self, file) -> Tuple[str, str, str, List, List]:
        """
        Handle PDF upload and complete processing
        
        Returns:
            Tuple of (status_md, pdf_info_md, text_preview, page_images, embedded_images)
        """
        try:
            if file is None:
                return (
                    "❌ **No file uploaded**", 
                    "", 
                    "", 
                    [], 
                    []
                )
            
            # Update status
            status_md = "⏳ **Processing PDF...**"
            
            # Process the PDF
            logger.info(f"Processing uploaded PDF: {file.name}")
            self.current_pdf_data = process_pdf(file.name)
            
            # Extract data for display
            pdf_info = self.current_pdf_data['pdf_info']
            text_data = self.current_pdf_data['text_data']
            page_images = self.current_pdf_data['page_images']
            embedded_images = self.current_pdf_data['embedded_images']
            summary = self.current_pdf_data['processing_summary']
            
            # Create status markdown
            status_md = f"""
## ✅ **PDF Processing Complete!**

### 📊 **Processing Summary:**
- **Pages Processed:** {summary['total_pages']}
- **Text Extracted:** {summary['total_words']:,} words ({summary['total_text_length']:,} characters)
- **Embedded Images:** {summary['embedded_images_count']} found
- **Potential Headshots:** {summary['potential_headshots']} detected
- **Status:** {'✅ Success' if summary['processing_success'] else '❌ Failed'}
"""
            
            # Create PDF info markdown
            pdf_info_md = f"""
## 📄 **PDF Information**

| Property | Value |
|----------|-------|
| **Filename** | {pdf_info['filename']} |
| **File Size** | {pdf_info['file_size_formatted']} |
| **Pages** | {pdf_info['page_count']} |
| **Title** | {pdf_info['title']} |
| **Author** | {pdf_info['author']} |
| **PDF Version** | {pdf_info['pdf_version']} |
| **Encrypted** | {'Yes' if pdf_info['encrypted'] else 'No'} |
| **Creation Date** | {pdf_info['creation_date']} |
"""
            
            # Create text preview (first 1000 characters)
            text_preview = text_data['total_text'][:1000]
            if len(text_data['total_text']) > 1000:
                text_preview += "\n\n... (truncated)"
            
            # Prepare page images for gallery - save to temp files with short names
            page_image_list = []
            for i, img in enumerate(page_images):
                # Save to temp file with short name
                temp_path = self._save_temp_image(img['base64'], f"page_{i+1}.png")
                page_image_list.append((temp_path, f"Page {img['page_number']} ({img['width']}x{img['height']})"))
            
            # Prepare embedded images for gallery - save to temp files with short names  
            embedded_image_list = []
            for i, img in enumerate(embedded_images):
                headshot_flag = "🖼️ Potential Headshot" if img['is_potential_headshot'] else "🖼️ Image"
                caption = f"{headshot_flag} - Page {img['page_number']} ({img['width']}x{img['height']}, {img['format']})"
                # Save to temp file with short name
                temp_path = self._save_temp_image(img['base64'], f"img_{i+1}.{img['format'].lower()}")
                embedded_image_list.append((temp_path, caption))
            
            logger.info(f"Successfully processed PDF: {pdf_info['filename']}")
            
            return (
                status_md,
                pdf_info_md, 
                text_preview,
                page_image_list,
                embedded_image_list
            )
            
        except PDFValidationError as e:
            error_msg = f"❌ **PDF Validation Failed**\n\n{str(e)}"
            logger.error(f"PDF validation failed: {str(e)}")
            return (error_msg, "", "", [], [])
            
        except PDFProcessingError as e:
            error_msg = f"❌ **PDF Processing Failed**\n\n{str(e)}"
            logger.error(f"PDF processing failed: {str(e)}")
            return (error_msg, "", "", [], [])
            
        except Exception as e:
            error_msg = f"❌ **Unexpected Error**\n\n{str(e)}"
            logger.error(f"Unexpected error: {str(e)}")
            return (error_msg, "", "", [], [])
    
    def _save_temp_image(self, base64_data: str, filename: str) -> str:
        """Save base64 image to temp file with short path"""
        import tempfile
        import base64
        
        # Create temp directory if needed with shorter path
        if not self.temp_image_dir:
            # Use C:\temp for Windows to avoid long paths
            try:
                import platform
                if platform.system() == "Windows":
                    base_temp = "C:\\temp"
                    os.makedirs(base_temp, exist_ok=True)
                    self.temp_image_dir = os.path.join(base_temp, "resume_imgs")
                    os.makedirs(self.temp_image_dir, exist_ok=True)
                else:
                    self.temp_image_dir = tempfile.mkdtemp(prefix="img_")
            except:
                # Fallback to system temp with shorter prefix
                self.temp_image_dir = tempfile.mkdtemp(prefix="img_")
        
        # Decode base64 image
        image_bytes = base64.b64decode(base64_data)
        
        # Save to temp file with short name
        temp_path = os.path.join(self.temp_image_dir, filename)
        
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        return temp_path
    
    def get_detailed_text_analysis(self) -> str:
        """Get detailed text analysis for the current PDF"""
        if not self.current_pdf_data:
            return "No PDF processed yet."
        
        text_data = self.current_pdf_data['text_data']
        
        analysis_md = "## 📝 **Detailed Text Analysis**\n\n"
        
        for page_data in text_data['pages']:
            analysis_md += f"""
### Page {page_data['page_number']}
- **Characters:** {page_data['char_count']:,}
- **Words:** {page_data['word_count']:,}
- **Text Blocks:** {page_data['blocks']}
- **Has Text:** {'Yes' if page_data['has_text'] else 'No'}

"""
        
        return analysis_md
    
    def get_image_analysis(self) -> str:
        """Get detailed image analysis for the current PDF"""
        if not self.current_pdf_data:
            return "No PDF processed yet."
        
        embedded_images = self.current_pdf_data['embedded_images']
        
        if not embedded_images:
            return "## 🖼️ **No embedded images found**"
        
        analysis_md = "## 🖼️ **Image Analysis**\n\n"
        
        for img in embedded_images:
            analysis_md += f"""
### Image {img['image_index'] + 1} (Page {img['page_number']})
- **Dimensions:** {img['width']} × {img['height']} pixels
- **Format:** {img['format']}
- **Size:** {format_file_size(img['size_bytes'])}
- **Aspect Ratio:** {img['aspect_ratio']:.2f}
- **Potential Headshot:** {'✅ Yes' if img['is_potential_headshot'] else '❌ No'}

"""
        
        return analysis_md
    
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
        
        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f, indent=2)
            return f.name

def create_interface():
    """Create the main Gradio interface"""
    
    app = ResumeImproverApp()
    
    with gr.Blocks(title="Resume Improver - PDF Testing", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🎯 Resume Improver - PDF Processing Test
        
        ## Phase 1: Foundation Testing
        Upload a PDF resume to test our PyMuPDF processing pipeline.
        
        **What we test:**
        - ✅ PDF upload and validation
        - ✅ Text extraction (all content)  
        - ✅ Page image conversion
        - ✅ Embedded image extraction
        - ✅ Headshot detection algorithm
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Upload Section
                gr.Markdown("## 📤 **Step 1: Upload PDF Resume**")
                
                pdf_file = gr.File(
                    label="Choose PDF Resume",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                process_btn = gr.Button(
                    "🔄 Process PDF", 
                    variant="primary",
                    size="lg"
                )
                
                # Status Display
                status_display = gr.Markdown(
                    "⏳ **Waiting for PDF upload...**",
                    label="Processing Status"
                )
            
            with gr.Column(scale=2):
                # PDF Info Section
                gr.Markdown("## 📊 **Step 2: PDF Information**")
                
                pdf_info_display = gr.Markdown(
                    "Upload a PDF to see file information...",
                    label="PDF Details"
                )
        
        # Results Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📝 **Step 3: Extracted Text Preview**")
                text_preview = gr.Textbox(
                    label="Text Content (First 1000 characters)",
                    lines=10,
                    max_lines=15,
                    interactive=False
                )
        
        # Images Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📄 **Step 4: PDF Page Images**")
                page_images_gallery = gr.Gallery(
                    label="Page Images",
                    show_label=True,
                    elem_id="page_images",
                    columns=2,
                    rows=2,
                    height="auto"
                )
            
            with gr.Column():
                gr.Markdown("## 🖼️ **Step 5: Embedded Images**")
                embedded_images_gallery = gr.Gallery(
                    label="Extracted Images (Potential Headshots)",
                    show_label=True,
                    elem_id="embedded_images",
                    columns=2,
                    rows=2,
                    height="auto"
                )
        
        # Analysis Tabs
        with gr.Tabs():
            with gr.TabItem("📊 Text Analysis"):
                text_analysis_btn = gr.Button("🔍 Get Detailed Text Analysis")
                text_analysis_display = gr.Markdown()
            
            with gr.TabItem("🖼️ Image Analysis"):
                image_analysis_btn = gr.Button("🔍 Get Detailed Image Analysis")
                image_analysis_display = gr.Markdown()
            
            with gr.TabItem("💾 Export Data"):
                export_btn = gr.Button("📥 Export Processing Data (JSON)")
                export_file = gr.File(label="Download Processing Data")
        
        # Event Handlers
        process_btn.click(
            fn=app.upload_and_process_pdf,
            inputs=[pdf_file],
            outputs=[
                status_display,
                pdf_info_display,
                text_preview,
                page_images_gallery,
                embedded_images_gallery
            ]
        )
        
        text_analysis_btn.click(
            fn=app.get_detailed_text_analysis,
            outputs=[text_analysis_display]
        )
        
        image_analysis_btn.click(
            fn=app.get_image_analysis,
            outputs=[image_analysis_display]
        )
        
        export_btn.click(
            fn=app.export_processing_data,
            outputs=[export_file]
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 🛠️ **Development Notes:**
        - Using PyMuPDF for all PDF operations
        - Phase 1: Testing PDF processing foundation
        - Next Phase: Add AI agents for resume analysis
        """)
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        debug=True             # Enable debug mode
    )