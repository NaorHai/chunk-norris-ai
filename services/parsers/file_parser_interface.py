from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List, Optional


class FileParserInterface(ABC):
    """
    Interface for file parsers that convert documents to markdown
    """
    
    @abstractmethod
    def parse(self, filepath: str) -> Union[str, Dict[str, Any]]:
        """
        Parse a file and convert it to markdown
        
        Args:
            filepath: Path to the file
            
        Returns:
            String containing markdown representation of the file or
            Dictionary with markdown and additional data
        """
        pass 