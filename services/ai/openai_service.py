"""
OpenAI service for refining markdown content and processing images.
"""
import requests
import json
import base64
import re
import sys
import os

# Add the project root directory to Python path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import the config, use a default key if not available
try:
    from config import OPENAI_API_KEY
except ImportError:
    print("Warning: config.py not found. Using a placeholder API key.")
    OPENAI_API_KEY = "your-openai-api-key-here"

class OpenAIService:
    """
    Service for interacting with OpenAI APIs.
    """
    
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        """
        Initialize the OpenAI service.
        
        Args:
            api_key: Optional API key, will use the one from config if not provided
            model: The model to use for refinement and image processing
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
    def _get_headers(self):
        """
        Get the headers for the API request.
        
        Returns:
            Dictionary containing the headers
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def refine_markdown(self, markdown_text):
        """
        Refine markdown content using OpenAI's model.
        
        Args:
            markdown_text: Raw markdown text to refine
            
        Returns:
            Refined markdown text with better structure and formatting
        """
        prompt = f"""
        Please improve the structure and formatting of the following markdown content extracted from a document. 
        Make sure to:
        1. Keep all important information
        2. Improve the formatting and structure
        3. Fix any typos or OCR errors if obvious
        4. Ensure all markdown syntax is correct and valid
        5. For any icons, symbols, or special characters that cannot be represented in plain text, describe them within square brackets (e.g., [checkmark icon], [arrow pointing right])
        6. Return ONLY the refined markdown without any additional explanations or metadata
        
        Here's the content to refine:
        
        {markdown_text}
        """
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that improves the structure and formatting of markdown content extracted from documents. Your output must be valid, properly formatted markdown. For any icons, symbols, or special characters that cannot be represented in plain text, describe them within square brackets."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.api_url, headers=self._get_headers(), json=payload)
            response.raise_for_status()  # Raise exception for non-200 status codes
            result = response.json()
            refined_markdown = result['choices'][0]['message']['content']
            
            # Validate and fix markdown format
            refined_markdown = self.validate_markdown_format(refined_markdown)
            
            return refined_markdown
        except Exception as e:
            print(f"Error refining markdown with OpenAI: {str(e)}")
            # Return original markdown if API call fails
            return markdown_text
    
    def process_image(self, image_path):
        """
        Process an image directly with OpenAI, bypassing OCR completely.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            String containing markdown output from OpenAI
        """
        # Read image as base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Create the payload with the image and prompt
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at analyzing document layouts, detecting reading order, and converting document content to well-structured markdown. When presented with an image of a document or text, analyze the layout carefully before responding.\n\nIMPORTANT: Your response MUST be valid markdown format. This includes:\n- Using proper heading levels with # syntax\n- Correctly formatted lists (ordered and unordered)\n- Proper table syntax with | and --- separators\n- Correct code blocks with ``` delimiters\n- Proper link and image syntax\n- No HTML tags unless absolutely necessary\n- For any icons, symbols, or special characters that cannot be represented in plain text, describe them within square brackets, e.g., [checkmark icon], [arrow pointing right], [company logo]"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is an image of a document or text. Please analyze the layout first, detecting whether it has columns, tables, sections, or other complex layouts. Then, extract all content and convert it to clean, well-structured markdown. Follow these steps:\n\n1. Analyze the layout (columns, reading order, tables, etc.)\n2. Extract the full text content\n3. Convert to properly structured markdown with appropriate headings, lists, tables, etc.\n4. Return ONLY the valid markdown output without additional explanations\n5. Ensure all markdown syntax is correct and properly formatted\n6. For any icons, symbols, or special characters that cannot be represented in plain text, describe them within square brackets (e.g., [checkmark icon], [arrow pointing right], etc.)"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.2
        }
        
        try:
            response = requests.post(self.api_url, headers=self._get_headers(), json=payload)
            response.raise_for_status()  # Raise exception for non-200 status codes
            result = response.json()
            markdown_content = result['choices'][0]['message']['content']
            
            # Validate markdown format
            markdown_content = self.validate_markdown_format(markdown_content)
            
            return markdown_content
        except Exception as e:
            print(f"Error processing image with OpenAI: {str(e)}")
            return f"Error processing image with AI: {str(e)}"
    
    def validate_markdown_format(self, markdown_text):
        """
        Validate and fix common markdown formatting issues.
        
        Args:
            markdown_text: The markdown text to validate
            
        Returns:
            Cleaned and validated markdown text
        """
        # Check if the response starts with explanatory text that should be removed
        lines = markdown_text.split('\n')
        start_idx = 0
        
        # Skip any introductory text before actual markdown content
        for i, line in enumerate(lines):
            if line.startswith('#') or line.startswith('-') or line.startswith('*') or line.startswith('1.') or line.startswith('|'):
                start_idx = i
                break
        
        # If we found a starting point for markdown content, remove preceding text
        if start_idx > 0:
            lines = lines[start_idx:]
            markdown_text = '\n'.join(lines)
        
        # Fix common markdown issues
        
        # Ensure proper heading spacing (# Heading, not #Heading)
        markdown_text = re.sub(r'(^|\n)#([^# ])', r'\1# \2', markdown_text)
        markdown_text = re.sub(r'(^|\n)##([^# ])', r'\1## \2', markdown_text)
        markdown_text = re.sub(r'(^|\n)###([^# ])', r'\1### \2', markdown_text)
        
        # Ensure proper list item spacing
        markdown_text = re.sub(r'(^|\n)-([^ ])', r'\1- \2', markdown_text)
        markdown_text = re.sub(r'(^|\n)\*([^ ])', r'\1* \2', markdown_text)
        
        # Ensure numbered lists have proper spacing
        markdown_text = re.sub(r'(^|\n)(\d+)\.([^ ])', r'\1\2. \3', markdown_text)
        
        # Remove any trailing "markdown" or "md" words that might be added
        markdown_text = re.sub(r'\n+markdown\s*$', '', markdown_text, flags=re.IGNORECASE)
        markdown_text = re.sub(r'\n+md\s*$', '', markdown_text, flags=re.IGNORECASE)
        
        # Remove explanatory comments that might be added
        markdown_text = re.sub(r'(?i)```markdown|```md', '```', markdown_text)
        
        # Look for unusual character sequences that might be icons and wrap them in square brackets if not already
        # This regex finds Unicode characters outside the basic Latin alphabet and common punctuation
        def replace_special_chars(match):
            char = match.group(0)
            # Skip if already within square brackets
            if re.search(r'\[[^\]]*' + re.escape(char) + r'[^\]]*\]', markdown_text):
                return char
            # Skip common characters we don't want to replace
            if char in '.,;:!?"\'-+=(){}[]<>/@#$%^&*_|\\~`' or char.isalnum() or char.isspace():
                return char
            # Replace with description in square brackets
            return f"[symbol: {char}]"
        
        # Apply the replacement for special characters
        # markdown_text = re.sub(r'[^\x00-\x7F]+', replace_special_chars, markdown_text)
        
        # Ensure all icon descriptions in square brackets are properly formatted
        # Look for incomplete brackets
        markdown_text = re.sub(r'\[[^\]]+$', lambda m: f"{m.group(0)}]", markdown_text)
        markdown_text = re.sub(r'^\][^\[]+', lambda m: f"[{m.group(0)[1:]}", markdown_text)
        
        return markdown_text 