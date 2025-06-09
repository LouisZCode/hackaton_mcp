# Resume Improver Multi-Agent System

## Overview
AI-powered resume improvement system using PyMuPDF for PDF processing and Claude's multi-modal capabilities for comprehensive content and visual analysis.

## Workflow
1. **Upload PDF** - User uploads resume PDF via Gradio interface
2. **PDF Processing** - Extract text, page images, embedded images, and metadata using PyMuPDF
3. **Multi-Modal AI Analysis** - Claude analyzes both content and visual presentation
4. **Results Delivery** - Comprehensive improvement suggestions with actionable recommendations

## Current Status: Phase 2 Complete ✅
**Multi-Modal AI Analysis System Ready for Testing**

### Phase 1 Features (Completed):
- ✅ PDF upload and validation with file size limits
- ✅ PyMuPDF unified processing (text + images + metadata)
- ✅ Enhanced Gradio interface with 5-step workflow
- ✅ Windows path compatibility and temp file handling
- ✅ Comprehensive error handling and validation

### Phase 2 Features (Completed):
- ✅ **Multi-Modal Analysis**: Text + visual design analysis using Claude vision
- ✅ **LangGraph Integration**: Sophisticated workflow with state management
- ✅ **LangFuse Observability**: Personalized trace naming using PDF author metadata
- ✅ **External Prompt Management**: Configurable prompts via prompts.yaml
- ✅ **Visual Analysis**: Layout, typography, spacing, and professional presentation feedback
- ✅ **Intelligent Fallback**: Automatic text-only mode when images unavailable

### Phase 3 (Planned):
- **Advanced Features**: Headshot analysis, ATS optimization, industry-specific recommendations
- **Content Generation**: Improved bullet points, cover letter generation
- **Enhanced Export**: PDF generation, Word export, before/after comparisons

## Setup

### Prerequisites
- Python 3.8+
- Virtual environment recommended

### Installation
```bash
# Install dependencies
uv add -r requirements.txt

# Copy environment template
cp .env.example .env

# Add your API keys to .env file
```

### Required API Keys
- **ANTHROPIC_API_KEY** (required) - Get from [Anthropic Console](https://console.anthropic.com/)
- **LANGFUSE_KEYS** (optional) - For observability and session tracking

## Usage

### Run the Application
```bash
python gradio_app.py
```
Application launches at `http://localhost:7860`

### Multi-Modal Analysis Workflow
1. **Upload PDF Resume** - Click "Choose PDF Resume" and select your file
2. **Process PDF** - Click "🔄 Process PDF" to extract text and images
3. **Analyze Resume** - Click "🚀 Analyze Resume with Claude" for AI analysis
4. **Review Results** - Get comprehensive feedback on both content and visual design
5. **Export Data** - Download complete analysis and suggestions

### Analysis Types
- **🎨 Multi-Modal Analysis**: When page images are available, Claude analyzes both content and visual presentation
- **📝 Text-Only Analysis**: Fallback mode focusing on content analysis when images unavailable

### Test with Sample PDFs
1. Add test resume PDFs to `test_resumes/` folder
2. Upload via Gradio interface  
3. Verify both text extraction and visual analysis work correctly

## Project Structure
```
resume_improver/
├── gradio_app.py              # Enhanced Gradio interface with multi-modal workflow
├── agent_template.py          # LangGraph multi-modal resume analyzer
├── pdf_processor.py           # PyMuPDF: unified PDF processing (text + images + metadata)
├── prompts.yaml               # External prompt management system
├── utils.py                   # Helper functions, validation & image processing config
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation (this file)
├── project_continuation.txt   # Comprehensive continuation guide for future development
└── test_resumes/              # Test PDFs folder
```

## Key Features

### Multi-Modal Analysis
- **Claude Vision Integration**: Analyzes both resume content and visual design
- **Intelligent Content Processing**: Text extraction with visual layout analysis
- **Professional Design Feedback**: Typography, spacing, visual hierarchy assessment

### Technical Architecture
- **LangGraph Workflows**: Sophisticated state management and error handling
- **LangFuse Observability**: Session tracking with personalized trace naming
- **External Configuration**: Prompts and settings managed via YAML files
- **Robust Error Handling**: Graceful fallback and comprehensive validation

### Performance Optimizations
- **Image Processing Limits**: Max 3 page images, 150 DPI, 5MB per image
- **Intelligent Fallback**: Automatic text-only mode when visual analysis unavailable
- **Windows Compatibility**: Resolved path length issues with proper temp directories

## Development Notes
- **PyMuPDF**: Unified approach for all PDF operations (text, images, page rendering)
- **Claude 3.7 Sonnet**: High-quality analysis with 6000 token limit for visual analysis
- **Modular Design**: Easy to extend with additional analysis agents and features
- **Configuration-Driven**: External prompts and settings for easy customization

## Next Steps
See `project_continuation.txt` for detailed roadmap and testing priorities.