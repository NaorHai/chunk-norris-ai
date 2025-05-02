from typing import Optional

from services.parsers.file_parser_interface import FileParserInterface


class TextParser(FileParserInterface):
    """
    Parser for plain text files
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the text parser
        
        Args:
            output_dir: Optional directory (not used for text parser)
        """
        self.output_dir = output_dir
        
    def parse(self, filepath: str) -> str:
        """
        Parse a plain text file and convert it to markdown format.
        
        Args:
            filepath: Path to the text file
            
        Returns:
            String containing markdown representation of the text
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Wrap code blocks in backticks for plain text
            markdown = f"```\n{content}\n```"
            return markdown
            
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(filepath, 'r', encoding='latin1') as f:
                    content = f.read()
                return f"```\n{content}\n```"
            except Exception as e:
                return f"Error parsing text file: {str(e)}"
        except Exception as e:
            return f"Error parsing text file: {str(e)}" 