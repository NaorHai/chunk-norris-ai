import os
import mimetypes
from typing import Dict, Any, Optional, Union

from config import OPENAI_API_KEY
from services.parsers import (
    FileParserInterface, 
    RTFParser, 
    PDFParser, 
    DOCXParser, 
    ImageParser, 
    TextParser
)
from services.ai import RTFAIProcessor
from services.parsers import (
    process_pdf_as_image_with_gpt4o, 
    process_image_with_gpt4o, 
    process_doc_as_image_with_gpt4o
)


class DocumentProcessorFactory:
    """
    Factory for creating document processors based on file type
    """
    
    @staticmethod
    def detect_file_type(filepath: str) -> str:
        """
        Detect the file type based on extension and MIME type.
        
        Args:
            filepath: Path to the file
            
        Returns:
            String representing the file type or extension
        """
        mime, _ = mimetypes.guess_type(filepath)
        file_ext = os.path.splitext(filepath)[1].lower()
        
        # Debug output
        print(f"File detection - MIME: {mime}, Extension: {file_ext}")
        
        # Prioritize extension for certain file types
        if file_ext == '.pdf':
            return 'application/pdf'
        elif file_ext in ['.doc', '.docx']:
            return 'application/msword'
        elif file_ext == '.rtf':
            return 'application/rtf'
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return 'image/' + file_ext[1:]
        
        # Fallback to MIME type or extension
        return mime or filepath.split('.')[-1]
    
    @staticmethod
    def create_parser(filepath: str, output_dir: Optional[str] = None) -> FileParserInterface:
        """
        Create a parser for the specified file type
        
        Args:
            filepath: Path to the file
            output_dir: Directory to save output files (like HTML previews)
            
        Returns:
            A parser instance that implements FileParserInterface
        """
        file_type = DocumentProcessorFactory.detect_file_type(filepath)
        file_ext = os.path.splitext(filepath)[1].lower()
        
        # Select parser based on file type
        if 'rtf' in file_type or file_ext == '.rtf':
            return RTFParser(output_dir=output_dir)
        elif 'pdf' in file_type or file_ext == '.pdf':
            return PDFParser(output_dir=output_dir)
        elif 'word' in file_type or file_ext in ['.docx', '.doc']:
            return DOCXParser(output_dir=output_dir)
        elif 'image' in file_type or file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return ImageParser(output_dir=output_dir)
        elif 'text' in file_type or file_ext in ['.txt', '.md', '.csv', '.json', '.xml', '.html', '.css', '.js']:
            return TextParser(output_dir=output_dir)
        else:
            # Fallback to text parser for unknown types
            print(f"No specific parser for file type {file_type} with extension {file_ext}, using TextParser")
            return TextParser(output_dir=output_dir)
    
    @staticmethod
    def create_ai_processor(filepath: str, output_dir: Optional[str] = None) -> Any:
        """
        Create an AI processor for the specified file type
        
        Args:
            filepath: Path to the file
            output_dir: Directory to save output files
            
        Returns:
            An AI processor instance appropriate for the file type
        """
        file_type = DocumentProcessorFactory.detect_file_type(filepath)
        file_ext = os.path.splitext(filepath)[1].lower()
        
        # Currently only RTF has a specialized AI processor
        # This would be extended in the future for other file types
        if 'rtf' in file_type or file_ext == '.rtf':
            return RTFAIProcessor(api_key=OPENAI_API_KEY, output_dir=output_dir)
        else:
            # Default AI processor for now
            return RTFAIProcessor(api_key=OPENAI_API_KEY, output_dir=output_dir)
    
    @staticmethod
    def process_file(filepath: str, use_ai: bool = False, output_dir: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Process a file using the appropriate parser or AI processor
        
        Args:
            filepath: Path to the file
            use_ai: Whether to use AI processing
            output_dir: Directory to save output files
            
        Returns:
            Markdown content or a dictionary with markdown and additional data
        """
        file_type = DocumentProcessorFactory.detect_file_type(filepath)
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if use_ai:
            processor = DocumentProcessorFactory.create_ai_processor(filepath, output_dir)
            
            # Handle files with specialized AI processing
            if 'rtf' in file_type or file_ext == '.rtf':
                return processor.process_rtf_with_gpt4o(filepath)
            elif 'pdf' in file_type or file_ext == '.pdf':
                # Use the existing PDF-to-image AI processing function
                print(f"Processing PDF with AI: {filepath}")
                markdown_content = process_pdf_as_image_with_gpt4o(filepath)
                return {"markdown": markdown_content, "html_preview": None}
            elif 'image' in file_type or file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                # Use the existing image AI processing function
                print(f"Processing image with AI: {filepath}")
                markdown_content = process_image_with_gpt4o(filepath)
                return {"markdown": markdown_content, "html_preview": None}
            elif 'word' in file_type or file_ext in ['.docx', '.doc']:
                # Use the existing DOCX-to-image AI processing function
                print(f"Processing DOCX with AI: {filepath}")
                markdown_content = process_doc_as_image_with_gpt4o(filepath)
                return {"markdown": markdown_content, "html_preview": None}
            else:
                # This would be extended to handle other file types with AI
                # For now, we'll inform the user that AI processing isn't available
                return {"markdown": f"AI processing not yet implemented for file type: {file_type}", "html_preview": None}
        else:
            parser = DocumentProcessorFactory.create_parser(filepath, output_dir)
            return parser.parse(filepath) 