"""
OCR interface defining the contract for all OCR implementations.
"""
from abc import ABC, abstractmethod

class OCRInterface(ABC):
    """
    Abstract base class for all OCR engines.
    Following the Strategy Pattern for different OCR implementations.
    """
    
    @abstractmethod
    def extract_text(self, image_path):
        """
        Extract text from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            String containing extracted text
        """
        pass
    
    @abstractmethod
    def supports_image_type(self, file_extension):
        """
        Check if this OCR engine supports a specific image type.
        
        Args:
            file_extension: The file extension to check
            
        Returns:
            Boolean indicating if the image type is supported
        """
        pass
    
    @property
    @abstractmethod
    def name(self):
        """
        Get the name of the OCR engine.
        
        Returns:
            String containing the name of the OCR engine
        """
        pass 