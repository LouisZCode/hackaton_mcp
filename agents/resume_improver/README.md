# Resume Improver Multi-Agent System

## Overview
AI-powered resume improvement system using PyMuPDF and Claude vision capabilities.

## Workflow
1. **Upload PDF** - User uploads resume PDF
2. **PDF Deconstruction** - Extract text, images, and metadata  
3. **AI Agent Analysis** - Multi-agent processing (async)
4. **Results Delivery** - Comprehensive improvement suggestions

## Current Phase: Foundation
Building core PDF processing and Gradio interface.

### Phase 1 Features (Current):
- ✅ PDF upload and validation
- ✅ PyMuPDF text extraction
- ✅ PyMuPDF image extraction  
- ✅ Basic Gradio interface
- ⏳ Workflow progress tracking

### Future Phases:
- **Phase 2**: AI agents (content analysis, headshot analysis)
- **Phase 3**: Result compilation and download

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
- **LANGFUSE_KEYS** (optional) - For observability

## Usage

### Run the Application
```bash
python gradio_app.py
```

### Test with Sample PDFs
1. Add test resume PDFs to `test_resumes/` folder
2. Upload via Gradio interface
3. Verify PDF processing works correctly

## Project Structure
```
resume_improver/
├── gradio_app.py              # Main interface + workflow
├── pdf_processor.py           # PyMuPDF: all PDF operations
├── utils.py                   # Helper functions & validation
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── .env.example              # Environment template
└── test_resumes/              # Test PDFs folder
```

## Development Notes
- Using PyMuPDF for all PDF operations (text, images, page rendering)
- Gradio for web interface with progress tracking
- Modular design for easy agent addition in later phases