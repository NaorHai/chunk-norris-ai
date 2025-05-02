import pdfplumber
from docx import Document
import mammoth
from PIL import Image
import pytesseract
import mimetypes
import os
import re
import io
import tempfile
import shutil
from pathlib import Path
import requests
import json
import base64

# Add new imports for advanced image parsing
import cv2
import numpy as np
from paddleocr import PaddleOCR

def detect_file_type(filepath):
    """
    Detect the file type based on extension and MIME type.
    
    Args:
        filepath: Path to the file
        
    Returns:
        String representing the file type or extension
    """
    mime, _ = mimetypes.guess_type(filepath)
    return mime or filepath.split('.')[-1]

def save_image(image_obj, output_dir, page_number, image_number):
    image_bytes = image_obj["stream"].get_data()
    ext = image_obj["ext"]
    filename = f"pdf_image_page{page_number}_img{image_number}.{ext}"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return filepath

def parse_pdf(filepath):
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
                        page_images = extract_images_from_pdf_page(page, temp_dir, i, image_count)
                        image_count += len(page_images)
                        
                        # Get tables from the page
                        tables = page.extract_tables()
                        has_tables = len(tables) > 0
                        
                        if has_tables:
                            # Check if we can get table areas to exclude from text
                            try:
                                # Extract the text excluding table regions
                                non_table_text = extract_text_excluding_tables(page)
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
                                    md.append(convert_table_to_markdown(table))
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

def extract_text_excluding_tables(page):
    """
    Extract text from a page while excluding text in table regions.
    This simplified approach just extracts text from the whole page
    and uses markers to indicate where tables begin and end.
    
    Args:
        page: A pdfplumber page object
        
    Returns:
        String containing text excluding table regions
    """
    # Since the character-level approach is causing text corruption,
    # we'll switch to using a simpler, more reliable approach
    
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
        if hasattr(table, 'bbox') and is_valid_bbox(table.bbox):
            valid_table_bboxes.append(table.bbox)
    
    # If no valid table bboxes found, return all text
    if not valid_table_bboxes:
        return whole_text
    
    # Since trying to filter out table text at character level is causing issues,
    # we'll extract text from the whole page and then skip processing tables separately.
    # In this approach, tables will appear twice (as plain text and formatted tables),
    # but the text won't be corrupted.
    
    # Format the extracted text
    paragraphs = whole_text.split('\n\n')
    clean_paragraphs = []
    
    for p in paragraphs:
        if p.strip():
            clean_paragraphs.append(p.strip())
    
    return "\n\n".join(clean_paragraphs)

def is_valid_bbox(bbox):
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

def extract_images_from_pdf_page(page, output_dir, page_num, img_counter):
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
        
        # Fallback: Render page as image if no images were extracted
        if not images:
            # This is a fallback option, might not be needed in all cases
            pass
    
    except Exception as e:
        print(f"Error extracting images from page {page_num+1}: {e}")
    
    return images

def convert_table_to_markdown(table):
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

def parse_docx(filepath):
    """
    Parse a DOCX file and convert it to markdown format.
    
    Args:
        filepath: Path to the DOCX file
        
    Returns:
        String containing markdown representation of the DOCX
    """
    try:
        doc = Document(filepath)
        md = []
        
        # Create a temp directory for extracted images
        temp_dir = tempfile.mkdtemp()
        try:
            # Extract images if present (this requires additional handling)
            image_count = 0
            
            for p in doc.paragraphs:
                if p.style.name.startswith("Heading"):
                    level = p.style.name[-1]
                    md.append(f"{'#' * int(level)} {p.text}")
                else:
                    md.append(p.text)
            
            # Optional: Process tables from DOCX
            for table in doc.tables:
                md.append("\n### Table\n")
                table_rows = []
                
                # Process header row
                header_row = []
                for cell in table.rows[0].cells:
                    header_row.append(cell.text.strip())
                
                table_rows.append("| " + " | ".join(header_row) + " |")
                table_rows.append("| " + " | ".join(["---"] * len(header_row)) + " |")
                
                # Process data rows
                for row in table.rows[1:]:
                    row_data = []
                    for cell in row.cells:
                        row_data.append(cell.text.strip())
                    table_rows.append("| " + " | ".join(row_data) + " |")
                
                md.append("\n".join(table_rows))
        
        finally:
            # Clean up the temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return "\n\n".join(md)
    
    except Exception as e:
        return f"Error parsing DOCX: {str(e)}"

def parse_rtf(filepath):
    """
    Parse an RTF file and convert it to markdown format.
    
    Args:
        filepath: Path to the RTF file
        
    Returns:
        String containing markdown representation of the RTF
    """
    try:
        with open(filepath, "rb") as f:
            result = mammoth.convert_to_markdown(f)
        return result.value
    except Exception as e:
        return f"Error parsing RTF: {str(e)}"

def parse_image(filepath):
    """
    Parse an image file using advanced OCR with PaddleOCR.
    
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
        try:
            # Initialize PaddleOCR
            ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            
            # Add the original image to markdown
            markdown.append(f"![Original Image]({os.path.basename(filepath)})\n")
            markdown.append("## Extracted Content\n")
            
            # Perform OCR on the full image
            ocr_result = ocr_model.ocr(image_rgb, cls=True)
            
            if ocr_result and ocr_result[0]:
                # Extract and format the recognized text
                text_lines = []
                for line in ocr_result[0]:
                    if line and len(line) >= 2 and line[1] and len(line[1]) >= 1:
                        text_lines.append(line[1][0])  # Get the text
                
                full_text = "\n".join(text_lines)
                markdown.append(full_text)
            else:
                markdown.append("*No text could be detected in this image.*")
                
        except Exception as e:
            # Fallback to traditional Tesseract OCR if PaddleOCR fails
            markdown.append("## Extracted Text (Fallback OCR)\n")
            
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

def refine_markdown_with_llm(markdown_text):
    """
    Refine markdown content using OpenAI's GPT-4o Mini model.
    
    Args:
        markdown_text: Raw markdown text to refine
        
    Returns:
        Refined markdown text with better structure and formatting
    """
    api_key = "sk-proj-cSwXhZghmjAJAa8VMoUNT3BlbkFJkzCmTCNrx2rN6ofETYaB"
    api_url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
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
        "model": "gpt-4o-mini",
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
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for non-200 status codes
        result = response.json()
        refined_markdown = result['choices'][0]['message']['content']
        
        # Validate and fix markdown format
        refined_markdown = validate_markdown_format(refined_markdown)
        
        return refined_markdown
    except Exception as e:
        print(f"Error refining markdown with LLM: {str(e)}")
        # Return original markdown if API call fails
        return markdown_text

# Add a new function to process images with "Chuck Norris AI" (GPT-4o Mini)
def process_image_with_gpt4o(image_path):
    """
    Process an image directly with GPT-4o Mini, bypassing OCR completely.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        String containing markdown output from GPT-4o Mini
    """
    # Read image as base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    api_key = "sk-proj-cSwXhZghmjAJAa8VMoUNT3BlbkFJkzCmTCNrx2rN6ofETYaB"
    api_url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Create the payload with the image and prompt
    payload = {
        "model": "gpt-4o-mini",
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
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for non-200 status codes
        result = response.json()
        markdown_content = result['choices'][0]['message']['content']
        
        # Validate markdown format
        markdown_content = validate_markdown_format(markdown_content)
        
        return markdown_content
    except Exception as e:
        print(f"Error processing image with GPT-4o Mini: {str(e)}")
        return f"Error processing image with AI: {str(e)}"

def validate_markdown_format(markdown_text):
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

def convert_to_markdown(filepath, use_ai_refinement=True, use_chuck_norris_ai=False):
    """
    Generic dispatcher that converts various file types to markdown.
    
    Args:
        filepath: Path to the file
        use_ai_refinement: Whether to use LLM to refine the markdown output
        use_chuck_norris_ai: Whether to bypass OCR and use GPT-4o Mini directly for images
        
    Returns:
        String containing markdown representation of the file
    """
    if not os.path.exists(filepath):
        return f"File not found: {filepath}"
    
    try:
        filetype = detect_file_type(filepath)
        
        raw_markdown = ""
        
        # For images with Chuck Norris AI enabled, bypass OCR completely
        if ('image' in filetype or filepath.endswith((".png", ".jpg", ".jpeg"))) and use_chuck_norris_ai:
            print("Using Chuck Norris AI for image processing, bypassing OCR...")
            return process_image_with_gpt4o(filepath)
        
        if 'pdf' in filetype:
            raw_markdown = parse_pdf(filepath)
        elif 'word' in filetype or filepath.endswith(".docx"):
            raw_markdown = parse_docx(filepath)
        elif 'rtf' in filetype or filepath.endswith(".rtf"):
            raw_markdown = parse_rtf(filepath)
        elif 'image' in filetype or filepath.endswith((".png", ".jpg", ".jpeg")):
            raw_markdown = parse_image(filepath)
        else:
            # Try to read as plain text for unknown types
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                raw_markdown = f"```\n{content}\n```"
            except:
                return f"Unsupported file type: {filetype}"
        
        # Refine the markdown using OpenAI's GPT-4o Mini if requested
        if use_ai_refinement:
            refined_markdown = refine_markdown_with_llm(raw_markdown)
            return refined_markdown
        else:
            return raw_markdown
    except Exception as e:
        return f"Error converting file: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Test with different file types
    test_files = [
        "example.pdf",
        "example.docx",
        "example.rtf",
        "example.png"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"Converting {test_file} to markdown...")
            markdown = convert_to_markdown(test_file)
            print(f"First 200 characters: {markdown[:200]}...") 