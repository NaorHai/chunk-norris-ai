import mimetypes
import os
from typing import Optional, Dict, Any

class FileDetector:
    """
    A class for detecting file types and extracting metadata from various document formats.
    Supports: PDF, DOCX, DOC, TXT, PNG, JPG, and more.
    """
    
    def __init__(self):
        self.mimetypes = mimetypes.MimeTypes()
        # Ensure common mimetypes are registered
        if not mimetypes.inited:
            mimetypes.init()
    
    def detect_file_type(self, filepath: str) -> str:
        """
        Detect the file type based on extension and content.
        
        Args:
            filepath: Path to the file
            
        Returns:
            String representing the file type or extension
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        # Get mimetype from extension
        mime, _ = self.mimetypes.guess_type(filepath)
        
        # If we couldn't determine the mime type, get it from the extension
        if not mime:
            extension = os.path.splitext(filepath)[1].lower()
            if extension:
                return extension[1:]  # Remove the dot
            else:
                return "unknown"
                
        return mime
    
    def get_file_metadata(self, filepath: str) -> Dict[str, Any]:
        """
        Get metadata for the file based on its type
        
        Args:
            filepath: Path to the file
            
        Returns:
            Dictionary containing metadata information
        """
        file_type = self.detect_file_type(filepath)
        metadata = {
            "filename": os.path.basename(filepath),
            "filepath": filepath,
            "size": os.path.getsize(filepath),
            "type": file_type,
        }
        
        return metadata
    
    def is_document(self, filepath: str) -> bool:
        """
        Check if the file is a document (PDF, DOCX, DOC, TXT)
        
        Args:
            filepath: Path to the file
            
        Returns:
            Boolean indicating if the file is a document
        """
        doc_types = ['application/pdf', 'application/msword', 
                     'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                     'text/plain']
        
        file_type = self.detect_file_type(filepath)
        
        # Check by mimetype or extension
        if file_type in doc_types:
            return True
        
        extension = os.path.splitext(filepath)[1].lower()
        if extension in ['.pdf', '.docx', '.doc', '.txt']:
            return True
            
        return False
    
    def is_image(self, filepath: str) -> bool:
        """
        Check if the file is an image
        
        Args:
            filepath: Path to the file
            
        Returns:
            Boolean indicating if the file is an image
        """
        file_type = self.detect_file_type(filepath)
        
        # Check by mimetype
        if file_type and file_type.startswith('image/'):
            return True
            
        # Check by extension
        extension = os.path.splitext(filepath)[1].lower()
        if extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']:
            return True
            
        return False


# Example usage
if __name__ == "__main__":
    detector = FileDetector()
    
    # Example file path
    test_file = "example.pdf"
    
    if os.path.exists(test_file):
        print(f"File type: {detector.detect_file_type(test_file)}")
        print(f"Metadata: {detector.get_file_metadata(test_file)}")
        print(f"Is document: {detector.is_document(test_file)}")
        print(f"Is image: {detector.is_image(test_file)}") 