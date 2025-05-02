"""
Tesseract OCR implementation.
"""
import pytesseract
from PIL import Image
from .ocr_interface import OCRInterface

class TesseractOCR(OCRInterface):
    """
    Tesseract OCR implementation.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Tesseract OCR engine.
        
        Args:
            config: Optional Tesseract configuration string
        """
        self._config = config or r'--oem 3 --psm 6'
    
    def extract_text(self, image_path):
        """
        Extract text from an image using Tesseract OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            String containing extracted text
        """
        try:
            # Load image and handle transparency if present
            img = Image.open(image_path)
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                img = background
            
            # Extract text with Tesseract OCR
            text = pytesseract.image_to_string(img, config=self._config)
            
            return text.strip() if text.strip() else "*No text could be extracted from this image.*"
        except Exception as e:
            return f"Error extracting text with Tesseract OCR: {str(e)}"
    
    def supports_image_type(self, file_extension):
        """
        Check if Tesseract OCR supports this image type.
        
        Args:
            file_extension: The file extension to check
            
        Returns:
            Boolean indicating if the image type is supported
        """
        supported_types = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif']
        return file_extension.lower() in supported_types
    
    @property
    def name(self):
        """
        Get the name of the OCR engine.
        
        Returns:
            String containing the name of the OCR engine
        """
        return "Tesseract OCR" 