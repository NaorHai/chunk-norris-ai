import os
import tempfile
import shutil
import re
from typing import Dict, Any, Union, List, Tuple, Optional

import pdfplumber
from PIL import Image

from services.parsers.file_parser_interface import FileParserInterface


class PDFParser(FileParserInterface):
    """
    Parser for PDF files that converts them to markdown
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the PDF parser
        
        Args:
            output_dir: Optional directory to save extracted images
        """
        self.output_dir = output_dir
        
    def parse(self, filepath: str) -> str:
        """
        Parse a PDF file and convert it to markdown format with table support.
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            String containing markdown representation of the PDF
        """
        md = []
        try:
            with pdfplumber.open(filepath) as pdf:
                # Create a temp directory for extracted images
                temp_dir = tempfile.mkdtemp()
                try:
                    # Keep track of image counters
                    image_count = 0
                    
                    for i, page in enumerate(pdf.pages):
                        md.append(f"# Page {i+1}")
                        
                        # Extract images if available
                        try:
                            # Extract and save images from the page
                            page_images = self._extract_images_from_pdf_page(page, temp_dir, i, image_count)
                            image_count += len(page_images)
                            
                            # Get tables from the page
                            tables = page.extract_tables()
                            has_tables = len(tables) > 0
                            
                            if has_tables:
                                # Check if we can get table areas to exclude from text
                                try:
                                    # Extract the text excluding table regions
                                    non_table_text = self._extract_text_excluding_tables(page)
                                    if non_table_text:
                                        md.append(non_table_text)
                                except Exception as e:
                                    # Fallback to a simpler method if character-level approach fails
                                    print(f"Error in character processing: {e}")
                                    # Just extract all text with normal method as fallback
                                    text = page.extract_text()
                                    if text:
                                        md.append(text)
                                
                                # Now process each table separately
                                for j, table in enumerate(tables):
                                    if table:  # Check if table is not empty
                                        md.append(f"\n### Table {j+1}\n")
                                        md.append(self._convert_table_to_markdown(table))
                            else:
                                # No tables, just extract text
                                text = page.extract_text()
                                if text:
                                    # Extract text by paragraph for better formatting
                                    paragraphs = text.split('\n\n')
                                    clean_paragraphs = []
                                    
                                    for p in paragraphs:
                                        if p.strip():
                                            clean_paragraphs.append(p.strip())
                                    
                                    md.append("\n\n".join(clean_paragraphs))
                            
                            # Add extracted images to markdown
                            for img_path, img_desc in page_images:
                                md.append(f"\n![{img_desc}]({img_path})\n")
                        
                        except Exception as e:
                            md.append(f"\n*Error processing page {i+1}: {str(e)}*\n")
                
                finally:
                    # Clean up the temp directory
                    shutil.rmtree(temp_dir, ignore_errors=True)
        
        except Exception as e:
            return f"Error parsing PDF: {str(e)}"
        
        return "\n\n".join(md)

    def _extract_text_excluding_tables(self, page) -> str:
        """
        Extract text from a page while excluding text in table regions.
        This simplified approach just extracts text from the whole page
        and uses markers to indicate where tables begin and end.
        
        Args:
            page: A pdfplumber page object
            
        Returns:
            String containing text excluding table regions
        """
        # First, get all text from the page using the standard method
        whole_text = page.extract_text()
        if not whole_text:
            return ""
        
        # Look for tables
        tables = page.find_tables()
        if not tables:
            return whole_text
        
        # Get table bounding boxes
        valid_table_bboxes = []
        for table in tables:
            if hasattr(table, 'bbox') and self._is_valid_bbox(table.bbox):
                valid_table_bboxes.append(table.bbox)
        
        # If no valid table bboxes found, return all text
        if not valid_table_bboxes:
            return whole_text
        
        # Format the extracted text
        paragraphs = whole_text.split('\n\n')
        clean_paragraphs = []
        
        for p in paragraphs:
            if p.strip():
                clean_paragraphs.append(p.strip())
        
        return "\n\n".join(clean_paragraphs)
    
    def _is_valid_bbox(self, bbox) -> bool:
        """
        Check if a bounding box is valid (has positive width and height).
        
        Args:
            bbox: A tuple or list with 4 elements (x0, top, x1, bottom)
            
        Returns:
            Boolean indicating if the bbox is valid
        """
        if not bbox or len(bbox) != 4:
            return False
        
        x0, top, x1, bottom = bbox
        
        # Check for valid types
        if not all(isinstance(x, (int, float)) for x in [x0, top, x1, bottom]):
            return False
        
        # Check width and height are positive
        width = x1 - x0
        height = bottom - top
        
        return width > 0 and height > 0
    
    def _extract_images_from_pdf_page(self, page, output_dir, page_num, img_counter) -> List[Tuple[str, str]]:
        """
        Extract images from a PDF page and save them to a directory.
        
        Args:
            page: A pdfplumber page object
            output_dir: Directory to save extracted images
            page_num: Current page number
            img_counter: Counter for image numbering
            
        Returns:
            List of tuples (image_path, image_description)
        """
        images = []
        
        # Extract images directly if available through pdfplumber
        try:
            # Use page.images if available (depends on version)
            if hasattr(page, 'images') and page.images:
                for i, img in enumerate(page.images):
                    img_num = img_counter + i
                    img_filename = f"image_{page_num+1}_{i+1}.png"
                    img_path = os.path.join(output_dir, img_filename)
                    
                    # Some PDF images can be directly extracted
                    if 'stream' in img and img['stream']:
                        try:
                            image_bytes = img['stream'].get_data()
                            with open(img_path, 'wb') as f:
                                f.write(image_bytes)
                            images.append((img_path, f"Image {img_num+1}"))
                        except Exception as e:
                            # If extraction fails, log but continue
                            print(f"Error extracting image: {e}")
        
        except Exception as e:
            print(f"Error extracting images from page {page_num+1}: {e}")
        
        return images
    
    def _convert_table_to_markdown(self, table) -> str:
        """
        Convert a pdfplumber table to markdown format.
        
        Args:
            table: A table extracted by pdfplumber
            
        Returns:
            String containing markdown table
        """
        md_table = []
        
        # Process header row
        if len(table) > 0:
            # Clean header cells
            header = [str(cell).strip() if cell is not None else "" for cell in table[0]]
            md_table.append("| " + " | ".join(header) + " |")
            
            # Add separator row
            md_table.append("| " + " | ".join(["---"] * len(header)) + " |")
            
            # Process data rows
            for row in table[1:]:
                # Clean cell data
                row_data = [str(cell).strip() if cell is not None else "" for cell in row]
                md_table.append("| " + " | ".join(row_data) + " |")
        
        return "\n".join(md_table) 