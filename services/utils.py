"""
Utility functions for the document-to-markdown converter application.
"""
import mimetypes
import os

def detect_file_type(filepath):
    """
    Detect the file type based on extension and MIME type.
    
    Args:
        filepath: Path to the file
        
    Returns:
        String representing the file type or extension
    """
    mime, _ = mimetypes.guess_type(filepath)
    return mime or os.path.splitext(filepath)[1].lstrip('.').lower() or 'unknown'

def format_filename(filename):
    """
    Format a filename to remove any invalid characters.
    
    Args:
        filename: The filename to format
        
    Returns:
        A formatted safe filename
    """
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(c for c in filename if c in valid_chars) 