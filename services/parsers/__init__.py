"""
Parser module for document and file parsing functionality.
Includes parsers for various file types including PDF, DOCX, RTF, Images, and more.
"""

from services.parsers.pdf_parser import PDFParser
from services.parsers.docx_parser import DOCXParser
from services.parsers.rtf_parser import RTFParser
from services.parsers.text_parser import TextParser
from services.parsers.image_parser import ImageParser
from services.parsers.file_detector import FileDetector
from services.parsers.file_parser_interface import FileParserInterface

# Import key functions from file_parsers
from services.parsers.file_parsers import (
    convert_to_markdown,
    parse_pdf,
    parse_docx,
    parse_rtf,
    parse_image,
    process_doc_with_gpt4o,
    process_pdf_as_image_with_gpt4o,
    process_image_with_gpt4o,
    process_doc_as_image_with_gpt4o,
    process_rtf_with_gpt4o,
    detect_file_type
)

__all__ = [
    'PDFParser',
    'DOCXParser',
    'RTFParser',
    'TextParser',
    'ImageParser',
    'FileDetector',
    'FileParserInterface',
    'convert_to_markdown',
    'parse_pdf',
    'parse_docx',
    'parse_rtf',
    'parse_image',
    'process_doc_with_gpt4o',
    'process_pdf_as_image_with_gpt4o',
    'process_image_with_gpt4o',
    'process_doc_as_image_with_gpt4o',
    'process_rtf_with_gpt4o',
    'detect_file_type'
] 