import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import pytesseract

from services.parsers.file_parser_interface import FileParserInterface


class ImageParser(FileParserInterface):
    """
    Parser for image files that extracts text using OCR
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the image parser
        
        Args:
            output_dir: Optional directory to save output files
        """
        self.output_dir = output_dir
        
    def parse(self, filepath: str) -> str:
        """
        Parse an image file using OCR with Tesseract.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            String containing markdown representation of the image with OCR text
        """
        try:
            # Step 1: Load image with OpenCV
            image = cv2.imread(filepath)
            if image is None:
                return f"Error loading image: {filepath}"
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            markdown = []
            
            # Add the original image to markdown
            markdown.append(f"![Original Image]({os.path.basename(filepath)})\n")
            markdown.append("## Extracted Content\n")
            
            # Use Tesseract OCR
            # Convert to RGB if image is in RGBA mode (for PNG transparency)
            img = Image.open(filepath)
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                img = background
            
            # Configure OCR for better results
            custom_config = r'--oem 3 --psm 6'  # Full page segmentation with OEM engine 3
            
            # Extract text with Tesseract OCR
            text = pytesseract.image_to_string(img, config=custom_config)
            
            if text.strip():
                markdown.append(text.strip())
            else:
                markdown.append("*No text could be extracted from this image.*")
            
            return "\n\n".join(markdown)
        
        except Exception as e:
            return f"Error processing image: {str(e)}" 