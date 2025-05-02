# Document-to-Markdown Converter

A web application that converts various document formats (PDF, DOCX, RTF, images) to Markdown.

## Features

- Converts PDF, DOCX, RTF, and image files to markdown
- Uses PaddleOCR for advanced OCR capabilities on images
- Optional AI refinement using OpenAI GPT-4o Mini
- "Chuck Norris AI" option to bypass OCR and directly process images with AI
- Interactive web interface with file preview and side-by-side comparison
- Rendered markdown preview option

## Setup

1. Clone the repository:
```bash
git clone git@github.com:NaorHai/rendering.git
cd rendering
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Setup API keys:
```bash
cp config.template.py config.py
# Edit config.py with your API keys
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:8090
```

## Architecture

The application follows SOLID principles with a modular design:

- `services/`: Service components organized by functionality
  - `ai/`: AI services for content refinement
  - `ocr/`: OCR engines for text extraction from images
  - `parsers/`: File format parsers
  - `converters/`: Format conversion logic

- `app.py`: Flask web application
- `templates/`: Web interface

## Requirements

- Python 3.8+
- Flask
- OpenAI API key (for AI refinement)
- PaddleOCR
- PyTesseract (with Tesseract OCR installed)

## License

MIT
