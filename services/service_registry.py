"""
Service registry for managing all service instances.
Implements a Singleton pattern for global access to services.
"""

from services.ai.openai_service import OpenAIService
from services.ocr.tesseract_ocr import TesseractOCR

class ServiceRegistry:
    """
    Registry for all services used in the application.
    Implements the Singleton pattern.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceRegistry, cls).__new__(cls)
            cls._instance._initialize_services()
        return cls._instance
    
    def _initialize_services(self):
        """Initialize all services"""
        self._services = {}
        
        # Initialize AI services
        self._services['openai'] = OpenAIService()
        
        # Initialize OCR services
        self._services['tesseract'] = TesseractOCR()
        
    def get_service(self, service_name):
        """
        Get a service by name.
        
        Args:
            service_name: Name of the service to get
            
        Returns:
            The service instance or None if not found
        """
        return self._services.get(service_name)
    
    def get_ai_service(self):
        """
        Get the AI service.
        
        Returns:
            The AI service instance
        """
        return self._services['openai']
    
    def get_ocr_service(self, service_name='tesseract'):
        """
        Get an OCR service.
        
        Args:
            service_name: Name of the OCR service to get
            
        Returns:
            The OCR service instance
        """
        return self._services.get(service_name) 