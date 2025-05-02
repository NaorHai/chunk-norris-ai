import os
import re
import uuid
import tempfile
import shutil
import base64
import requests
from typing import Dict, Any, List, Optional, Union

from services.parsers.rtf_parser import RTFParser
from services.ai.html_renderer import HtmlRenderer
from services.config import GPT4O_MINI_MODEL, DEFAULT_TEMPERATURE, MAX_CHUNK_SIZE, RTF_PREVIEWS_DIR


class RTFAIProcessor:
    """
    Process RTF files using GPT-4o Mini AI
    """
    
    def __init__(self, api_key: str, output_dir: str = None):
        """
        Initialize the RTF AI processor
        
        Args:
            api_key: OpenAI API key
            output_dir: Directory to save HTML previews and extracted images
        """
        self.api_key = api_key
        self.output_dir = output_dir or RTF_PREVIEWS_DIR
        self.rtf_parser = RTFParser(output_dir=self.output_dir)
    
    def process_rtf_with_gpt4o(self, rtf_path: str) -> Dict[str, Any]:
        """
        Process an RTF file with GPT-4o Mini, extracting embedded images if available.
        Creates an HTML representation for display in the UI.
        
        Args:
            rtf_path: Path to the RTF file
            
        Returns:
            Dictionary containing markdown output and HTML preview path
        """
        try:
            print(f"Processing RTF file: {os.path.basename(rtf_path)}")
            
            # Check file size first to avoid token limit issues
            file_size = os.path.getsize(rtf_path)
            if file_size > 5 * 1024 * 1024:  # 5MB limit
                print(f"RTF file too large ({file_size / (1024*1024):.2f} MB), chunking content...")
                return self.process_large_rtf_file(rtf_path)
            
            # Create a temporary directory for extracted images and HTML
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Convert RTF to HTML for display and processing
                html_path, extracted_images = self.rtf_parser.rtf_to_html(rtf_path, temp_dir)
                
                # Create a unique name for the final HTML that will be preserved
                final_html_dir = self.output_dir
                os.makedirs(final_html_dir, exist_ok=True)
                
                final_html_name = f"rtf_preview_{uuid.uuid4().hex}.html"
                final_html_path = os.path.join(final_html_dir, final_html_name)
                
                # Copy the HTML file to a location that can be accessed by the browser
                shutil.copy2(html_path, final_html_path)
                
                # Copy any extracted images to the same directory
                for img_path in extracted_images:
                    img_filename = os.path.basename(img_path)
                    shutil.copy2(img_path, os.path.join(final_html_dir, img_filename))
                
                # Generate an image from HTML for LLM processing
                html_image_path = os.path.join(temp_dir, f"rtf_as_image_{uuid.uuid4().hex}.png")
                renderer = HtmlRenderer()
                rendered_image_path = renderer.html_to_image(html_path, html_image_path)
                
                # If HTML to image conversion succeeded, use that for LLM
                if rendered_image_path and os.path.exists(rendered_image_path):
                    print("Successfully rendered HTML to image for LLM processing")
                    markdown_content = self._process_image_with_gpt4o(rendered_image_path)
                else:
                    print("HTML rendering failed, falling back to text extraction")
                    # Fallback to text extraction - don't use mammoth as it expects ZIP files
                    try:
                        # Try to extract plain text from RTF
                        try:
                            import striprtf.striprtf as striprtf
                            with open(rtf_path, 'rb') as rtf_file:
                                rtf_text = rtf_file.read().decode('utf-8', errors='ignore')
                                plain_text = striprtf.rtf_to_text(rtf_text)
                        except ImportError:
                            # Basic RTF to text
                            with open(rtf_path, 'r', encoding='utf-8', errors='ignore') as f:
                                rtf_text = f.read()
                                # Remove RTF headers and control words
                                plain_text = re.sub(r'\\[a-z0-9]+', ' ', rtf_text)
                                plain_text = re.sub(r'\{|\}|\\', '', plain_text)
                        
                        # Truncate if too large
                        if len(plain_text) > 50000:
                            print(f"RTF text content too large ({len(plain_text)} chars), truncating...")
                            plain_text = plain_text[:25000] + "\n\n... [Content truncated due to size] ...\n\n" + plain_text[-25000:]
                        
                        # Process text with GPT-4o
                        markdown_content = self._process_text_with_gpt4o(plain_text)
                    except Exception as text_err:
                        print(f"Error processing RTF as text: {str(text_err)}")
                        markdown_content = f"Error processing RTF file: {str(text_err)}"
                
                # Process a sample of extracted images separately if available
                image_markdown = []
                if extracted_images:
                    image_markdown.append("\n## Embedded Images\n")
                    
                    for i, img_path in enumerate(extracted_images[:3]):  # Limit to 3 images
                        try:
                            # Process the image with GPT-4o
                            image_result = self._process_image_with_gpt4o(img_path)
                            image_markdown.append(f"\n### Embedded Image {i+1}\n")
                            image_markdown.append(image_result)
                        except Exception as img_err:
                            print(f"Error processing embedded image {i+1}: {str(img_err)}")
                            image_markdown.append(f"\n*Error processing embedded image {i+1}: {str(img_err)}*\n")
                    
                    if len(extracted_images) > 3:
                        image_markdown.append(f"\n*{len(extracted_images) - 3} additional images were found but not processed.*\n")
                
                # Combine text and image results
                combined_markdown = markdown_content
                if image_markdown:
                    combined_markdown += "\n\n" + "\n\n".join(image_markdown)
                
                # Return both the markdown and the HTML preview path
                return {
                    'markdown': combined_markdown,
                    'html_preview': f"/static/rtf_previews/{final_html_name}"
                }
                    
            finally:
                # Clean up the temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        except Exception as e:
            print(f"Error processing RTF file: {str(e)}")
            import traceback
            print(f"Exception traceback: {traceback.format_exc()}")
            return {
                'markdown': f"Error processing RTF file: {str(e)}",
                'html_preview': None
            }
    
    def process_large_rtf_file(self, rtf_path: str) -> Dict[str, Any]:
        """
        Process a large RTF file by breaking it into manageable chunks.
        Also creates an HTML representation for display in the UI.
        
        Args:
            rtf_path: Path to the RTF file
            
        Returns:
            Dictionary containing markdown output and HTML preview path
        """
        try:
            # Create a temp directory for extracted images
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Convert RTF to HTML for display
                html_path, extracted_images = self.rtf_parser.rtf_to_html(rtf_path, temp_dir)
                
                # Create a unique name for the final HTML that will be preserved
                final_html_dir = self.output_dir
                os.makedirs(final_html_dir, exist_ok=True)
                
                final_html_name = f"rtf_preview_large_{uuid.uuid4().hex}.html"
                final_html_path = os.path.join(final_html_dir, final_html_name)
                
                # Copy the HTML file to a location that can be accessed by the browser
                shutil.copy2(html_path, final_html_path)
                
                # Copy any extracted images to the same directory
                for img_path in extracted_images:
                    img_filename = os.path.basename(img_path)
                    shutil.copy2(img_path, os.path.join(final_html_dir, img_filename))
                
                # Extract text directly from RTF, not using mammoth
                try:
                    # Try to extract plain text from RTF
                    try:
                        import striprtf.striprtf as striprtf
                        with open(rtf_path, 'rb') as rtf_file:
                            rtf_text = rtf_file.read().decode('utf-8', errors='ignore')
                            text_content = striprtf.rtf_to_text(rtf_text)
                    except ImportError:
                        # Basic RTF to text
                        with open(rtf_path, 'r', encoding='utf-8', errors='ignore') as f:
                            rtf_text = f.read()
                            # Remove RTF headers and control words
                            text_content = re.sub(r'\\[a-z0-9]+', ' ', rtf_text)
                            text_content = re.sub(r'\{|\}|\\', '', text_content)
                except Exception as e:
                    print(f"Error extracting text from large RTF: {str(e)}")
                    # If all text extraction fails, create placeholder text
                    text_content = "[The RTF content could not be extracted properly]"
                
                # Split text into chunks of approximately MAX_CHUNK_SIZE characters
                chunks = [text_content[i:i+MAX_CHUNK_SIZE] for i in range(0, len(text_content), MAX_CHUNK_SIZE)]
                
                # Process each chunk and collect results
                processed_chunks = []
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        # Process first chunk normally
                        try:
                            chunk_result = self._process_text_with_gpt4o(f"This is the first part of a document that has been split into chunks due to size limitations.\n\n{chunk}")
                            processed_chunks.append(chunk_result)
                        except Exception as chunk_err:
                            print(f"Error processing chunk {i+1}: {str(chunk_err)}")
                            processed_chunks.append(f"*Error processing chunk {i+1}*\n\n{chunk[:500]}...\n")
                    elif i == len(chunks) - 1:
                        # Process last chunk
                        try:
                            chunk_result = self._process_text_with_gpt4o(f"This is the last part of a document that has been split into chunks due to size limitations.\n\n{chunk}")
                            processed_chunks.append(chunk_result)
                        except Exception as chunk_err:
                            print(f"Error processing chunk {i+1}: {str(chunk_err)}")
                            processed_chunks.append(f"*Error processing chunk {i+1}*\n\n{chunk[:500]}...\n")
                    else:
                        # Process middle chunks
                        try:
                            chunk_result = self._process_text_with_gpt4o(f"This is part {i+1} of a document that has been split into chunks due to size limitations.\n\n{chunk}")
                            processed_chunks.append(chunk_result)
                        except Exception as chunk_err:
                            print(f"Error processing chunk {i+1}: {str(chunk_err)}")
                            processed_chunks.append(f"*Error processing chunk {i+1}*\n\n{chunk[:500]}...\n")
                
                # Process up to 3 images
                image_results = []
                for i, img_path in enumerate(extracted_images[:3]):
                    try:
                        image_result = self._process_image_with_gpt4o(img_path)
                        image_results.append(f"\n### Embedded Image {i+1}\n\n{image_result}")
                    except Exception as img_err:
                        print(f"Error processing image {i+1}: {str(img_err)}")
                
                # Combine results
                combined_markdown = "\n\n# Document Content (Processed in Chunks)\n\n"
                for i, chunk_result in enumerate(processed_chunks):
                    combined_markdown += f"\n\n## Part {i+1}\n\n{chunk_result}\n\n"
                
                if image_results:
                    combined_markdown += "\n\n# Embedded Images\n\n"
                    combined_markdown += "\n\n".join(image_results)
                
                if len(extracted_images) > 3:
                    combined_markdown += f"\n\n*Note: {len(extracted_images) - 3} additional images were found but not processed due to size limitations.*"
                
                return {
                    'markdown': combined_markdown,
                    'html_preview': f"/static/rtf_previews/{final_html_name}"
                }
                
            finally:
                # Clean up
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        except Exception as e:
            print(f"Error processing large RTF file: {str(e)}")
            return {
                'markdown': f"Error processing large RTF file: {str(e)}",
                'html_preview': None
            }

    def _process_image_with_gpt4o(self, image_path: str) -> str:
        """
        Process an image file directly with GPT-4o Mini.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            String containing markdown output from GPT-4o Mini
        """
        try:
            # Read image as base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Get file extension for content type
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext in ['.jpg', '.jpeg']:
                content_type = "image/jpeg"
            elif file_ext == '.png':
                content_type = "image/png"
            elif file_ext == '.gif':
                content_type = "image/gif"
            elif file_ext == '.bmp':
                content_type = "image/bmp"
            else:
                content_type = "image/jpeg"  # Default
            
            print(f"Processing image file: {os.path.basename(image_path)}")
            return self._send_image_to_gpt4o(base64_image, content_type, image_path)
        
        except Exception as e:
            print(f"Error processing image file: {str(e)}")
            return f"Error processing image file: {str(e)}"
    
    def _process_text_with_gpt4o(self, text_content: str) -> str:
        """
        Process text content with GPT-4o Mini.
        
        Args:
            text_content: The text content to process
            
        Returns:
            String containing markdown output from GPT-4o Mini
        """
        api_url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        system_prompt = "You are an expert at converting document content to well-structured markdown. Your output must be valid, properly formatted markdown."
        
        user_prompt = f"""
        Please convert the following content to clean, well-structured markdown. 
        Make sure to:
        1. Keep all important information
        2. Use proper markdown structure with headings, lists, tables, etc.
        3. Ensure all markdown syntax is correct and valid
        4. Return ONLY the converted markdown without additional explanations
        
        Here's the content to convert:
        
        {text_content}
        """
        
        payload = {
            "model": GPT4O_MINI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": DEFAULT_TEMPERATURE
        }
        
        try:
            print("\n=== SENDING LLM REQUEST (Text Processing) ===")
            print(f"Model: {GPT4O_MINI_MODEL}")
            print(f"Text length: {len(text_content)} characters")
            print(f"Temperature: {DEFAULT_TEMPERATURE}")
            
            response = requests.post(api_url, headers=headers, json=payload)
            
            # More detailed error handling
            if response.status_code != 200:
                error_msg = f"API Error: {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f" - {error_data['error'].get('message', 'Unknown error')}"
                        if 'code' in error_data['error']:
                            error_msg += f" (Code: {error_data['error']['code']})"
                        if 'param' in error_data['error']:
                            error_msg += f" - Parameter: {error_data['error']['param']}"
                    print(f"Full error response: {error_data}")
                except Exception as json_err:
                    error_msg += f" - {response.text[:200]}..."
                
                raise Exception(error_msg)
                
            response.raise_for_status()
            result = response.json()
            
            print(f"Response received, status code: {response.status_code}")
            print(f"Tokens used: {result.get('usage', {}).get('total_tokens', 'N/A')}")
            print("=== END OF LLM REQUEST ===\n")
            
            markdown_content = result['choices'][0]['message']['content']
            
            # Validate markdown format
            markdown_content = self._validate_markdown_format(markdown_content)
            
            return markdown_content
        except Exception as e:
            print(f"Error processing text with GPT-4o Mini: {str(e)}")
            # For debugging only, don't include in production
            import traceback
            print(f"Exception traceback: {traceback.format_exc()}")
            return f"Error processing text with AI: {str(e)}"
    
    def _send_image_to_gpt4o(self, base64_data: str, content_type: str, filepath: str) -> str:
        """
        Send an image or document to GPT-4o Mini and get markdown output.
        
        Args:
            base64_data: Base64 encoded image data
            content_type: MIME type of the content
            filepath: Original file path (for reporting)
            
        Returns:
            String containing markdown output from GPT-4o Mini
        """
        api_url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        print(f"Using Chuck Norris AI for document processing, content type: {content_type}")
        
        # Load system and user prompts from mustache template file
        system_prompt = ""
        user_prompt = ""
        
        try:
            with open('prompts/gpt4o_flow_chart_prompt.mustache', 'r') as template_file:
                template_content = template_file.read()
                
                # Extract system prompt (between first {{! System prompt... }} and {{! User prompt... }})
                system_start = template_content.find("{{! System prompt")
                user_start = template_content.find("{{! User prompt")
                
                if system_start != -1 and user_start != -1:
                    # Extract system prompt (skip the comment line)
                    system_prompt_with_comment = template_content[system_start:user_start].strip()
                    system_prompt_lines = system_prompt_with_comment.split('\n')
                    system_prompt = '\n'.join(system_prompt_lines[1:]).strip()
                    
                    # Extract user prompt (skip the comment line)
                    user_prompt_with_comment = template_content[user_start:].strip()
                    user_prompt_lines = user_prompt_with_comment.split('\n')
                    user_prompt = '\n'.join(user_prompt_lines[1:]).strip()
        except Exception as e:
            print(f"Error loading prompt template: {str(e)}, using fallback prompts")
            # Fallback to hardcoded prompts
            system_prompt = "You are an expert at analyzing document layouts, detecting reading order, and converting document content to well-structured markdown. When presented with an image of a document or text, analyze the layout carefully before responding.\n\nIMPORTANT: Your response MUST be valid markdown format. This includes:\n- Using proper heading levels with # syntax\n- Correctly formatted lists (ordered and unordered)\n- Proper table syntax with | and --- separators\n- Correct code blocks with ``` delimiters\n- Proper link and image syntax\n- No HTML tags unless absolutely necessary\n- For any icons, symbols, or special characters that cannot be represented in plain text, describe them within square brackets, e.g., [checkmark icon], [arrow pointing right], [company logo]"
            user_prompt = "This is a document. Please analyze the layout first, detecting whether it has columns, tables, sections, flow charts, or other complex layouts. Then, extract all content and convert it to clean, well-structured markdown. Follow these steps:\n\n1. Analyze the layout (columns, reading order, tables, flow charts, etc.)\n2. Extract the full text content\n3. Convert to properly structured markdown with appropriate headings, lists, tables, etc.\n4. Return ONLY the valid markdown output without additional explanations\n5. Ensure all markdown syntax is correct and properly formatted\n6. For any icons, symbols, or special characters that cannot be represented in plain text, describe them within square brackets (e.g., [checkmark icon], [arrow pointing right], etc.)\n\nIMPORTANT - FOR FLOW CHARTS AND DIAGRAMS:\n- If the document contains a flow chart or diagram, represent its structure in markdown\n- For each vertex/node in the diagram, provide a concise one-sentence description\n- For each relationship (edge/arc) between vertices, explicitly list as 'Vertex A → Vertex B' (showing direction with an arrow)\n- Indicate whether relationships are one-directional (→) or bi-directional (↔) between nodes\n- List all vertex-to-vertex relationships to completely map the structure\n- Include a clear section titled '## Graph Structure' that enumerates all relationships\n- After listing all relationships, include a single sentence summarizing the overall flow or purpose\n- Do not just extract the text without capturing the structural relationships and direction of flow"
        
        # Create the payload with the document and prompt
        user_content = [
            {
                "type": "text",
                "text": user_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{content_type};base64,{base64_data}"
                }
            }
        ]
        
        payload = {
            "model": GPT4O_MINI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "temperature": DEFAULT_TEMPERATURE
        }
        
        try:
            print("\n=== SENDING LLM REQUEST (Document Processing) ===")
            print(f"Model: {GPT4O_MINI_MODEL}")
            print(f"Document: {os.path.basename(filepath)}")
            print(f"File size: {os.path.getsize(filepath)} bytes")
            print(f"Content type: {content_type}")
            print(f"Temperature: {DEFAULT_TEMPERATURE}")
            
            response = requests.post(api_url, headers=headers, json=payload)
            
            # More detailed error handling
            if response.status_code != 200:
                error_msg = f"API Error: {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f" - {error_data['error'].get('message', 'Unknown error')}"
                        if 'code' in error_data['error']:
                            error_msg += f" (Code: {error_data['error']['code']})"
                        if 'param' in error_data['error']:
                            error_msg += f" - Parameter: {error_data['error']['param']}"
                    print(f"Full error response: {error_data}")
                except Exception as json_err:
                    error_msg += f" - {response.text[:200]}..."
                
                raise Exception(error_msg)
                
            response.raise_for_status()  # This will only run if status_code is 200
            result = response.json()
            
            print(f"Response received, status code: {response.status_code}")
            print(f"Tokens used: {result.get('usage', {}).get('total_tokens', 'N/A')}")
            print("=== END OF LLM REQUEST ===\n")
            
            markdown_content = result['choices'][0]['message']['content']
            
            # Validate markdown format
            markdown_content = self._validate_markdown_format(markdown_content)
            
            return markdown_content
        except Exception as e:
            print(f"Error processing document with GPT-4o Mini: {str(e)}")
            # For debugging only, don't include in production
            import traceback
            print(f"Exception traceback: {traceback.format_exc()}")
            return f"Error processing document with AI: {str(e)}"
    
    def _validate_markdown_format(self, markdown_text: str) -> str:
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
        
        # Count backtick groups to ensure balanced code blocks
        backtick_groups = re.findall(r'```', markdown_text)
        if len(backtick_groups) % 2 != 0:
            # Unbalanced code blocks, add closing backticks if needed
            if markdown_text.strip().endswith('```'):
                # Remove trailing backticks if they're at the end
                markdown_text = re.sub(r'```\s*$', '', markdown_text.strip())
            else:
                # Add closing backticks if we end with an open code block
                markdown_text = markdown_text.strip() + '\n```'
        
        # Ensure all icon descriptions in square brackets are properly formatted
        # Look for incomplete brackets
        markdown_text = re.sub(r'\[[^\]]+$', lambda m: f"{m.group(0)}]", markdown_text)
        markdown_text = re.sub(r'^\][^\[]+', lambda m: f"[{m.group(0)[1:]}", markdown_text)
        
        return markdown_text 