# Document-to-Markdown Converter

A Flask application that converts various document types (PDF, DOCX, RTF, images) to markdown format using both traditional OCR and AI-powered approaches.

## Features

- Convert PDF, DOCX, RTF, and image files to markdown
- RTF file specialized handling:
  - Converts RTF to HTML for display in the left panel
  - Processes RTF content with LLM in the right panel
- Two processing modes:
  - Traditional OCR for text extraction
  - "Chuck Norris AI" that leverages OpenAI's GPT-4o Mini for intelligent content extraction

## Modular Architecture

The application follows SOLID principles with the following structure:

```
.
├── app.py                 # Main Flask application
├── config.py              # API keys and global configuration
├── services/              # Modular services
│   ├── ai/                # AI processing services
│   │   ├── html_renderer.py     # HTML to image rendering for AI
│   │   └── rtf_ai_processor.py  # RTF-specific AI processing
│   ├── config.py          # Service configuration and constants
│   ├── factory.py         # Factory for creating appropriate processors
│   ├── parsers/           # Document parsers
│   │   ├── file_parser_interface.py  # Interface for file parsers
│   │   └── rtf_parser.py            # RTF-specific parser
│   └── service_registry.py  # Service registry
├── static/               # Static files
│   └── rtf_previews/     # Generated HTML previews for RTF files
├── templates/            # Flask templates
│   └── index.html        # Main application UI
└── uploads/              # Uploaded files
    └── images/           # Extracted and uploaded images
```

## RTF File Processing Flow

1. RTF files are parsed to extract text and embedded images
2. The RTF is converted to HTML for display in the left panel
3. For AI processing:
   - HTML is rendered to an image
   - Image is sent to GPT-4o Mini for intelligent parsing
   - Embedded images are processed separately
4. The results are combined into a comprehensive markdown document

## API Usage

### Upload Endpoint

```
POST /upload
```

**Parameters:**

- `file`: The file to convert (multipart/form-data)
- `use_ai_refinement`: Whether to refine the output with AI (default: true)
- `use_chuck_norris_ai`: Whether to use GPT-4o Mini for processing (default: false)

**Response:**

```json
{
  "success": true,
  "filename": "original_filename.rtf",
  "markdown": "# Converted markdown content...",
  "html_preview": "/static/rtf_previews/rtf_preview_abcdef123456.html",
  "used_ai_refinement": true,
  "used_chuck_norris_ai": true,
  "is_rtf": true
}
```

## Running the Application

1. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

2. Copy `config.template.py` to `config.py` and add your OpenAI API key.

3. Run the application:
   ```
   python app.py
   ```

4. The application will be available at `http://localhost:8080`. 