import os
import re
import uuid
import tempfile
import shutil
import html
from typing import Dict, Any, Union, List, Tuple, Optional
from pathlib import Path
from bs4 import BeautifulSoup

from services.parsers.file_parser_interface import FileParserInterface
from services.config import RTF_PREVIEWS_DIR


class RTFParser(FileParserInterface):
    """
    Parser for RTF files that converts them to markdown and HTML
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the RTF parser
        
        Args:
            output_dir: Directory to save HTML previews and extracted images
        """
        self.output_dir = output_dir or RTF_PREVIEWS_DIR
        
    def parse(self, filepath: str) -> Dict[str, Any]:
        """
        Parse an RTF file and convert it to markdown
        Extract both text content and embedded images.
        
        Args:
            filepath: Path to the RTF file
            
        Returns:
            Dictionary containing markdown representation and HTML preview path
        """
        try:
            # First try to extract text with mammoth
            from mammoth import convert_to_markdown
            with open(filepath, "rb") as f:
                result = convert_to_markdown(f)
            
            text_content = result.value
            
            # Create a temporary directory for extracted images
            temp_dir = tempfile.mkdtemp()
            extracted_images = []
            
            try:
                # Extract embedded images using a more robust approach
                extracted_images = self.extract_images_from_rtf(filepath, temp_dir)
                
                # Convert to HTML for preview
                html_path, extracted_images = self.rtf_to_html(filepath, temp_dir)
                
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
                
                # Combine text and image references into final markdown
                markdown_lines = []
                
                # Add the text content first
                if text_content.strip():
                    markdown_lines.append(text_content.strip())
                
                # Add extracted images
                if extracted_images:
                    markdown_lines.append("\n## Embedded Images\n")
                    for i, img_path in enumerate(extracted_images):
                        img_filename = os.path.basename(img_path)
                        markdown_lines.append(f"\n![Embedded Image {i+1}]({img_filename})\n")
                
                markdown_content = "\n\n".join(markdown_lines)
                
                # Return both the markdown and the HTML preview path
                return {
                    'markdown': markdown_content,
                    'html_preview': f"/static/rtf_previews/{final_html_name}"
                }
                
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception as e:
            print(f"Error parsing RTF file: {str(e)}")
            import traceback
            print(f"Exception traceback: {traceback.format_exc()}")
            return {
                'markdown': f"Error parsing RTF: {str(e)}",
                'html_preview': None
            }
    
    def extract_images_from_rtf(self, rtf_path: str, output_dir: str) -> List[str]:
        """
        Extract embedded images from an RTF file.
        
        Args:
            rtf_path: Path to the RTF file
            output_dir: Directory to save extracted images
            
        Returns:
            List of paths to extracted images
        """
        extracted_images = []
        
        try:
            # Read the RTF file as binary
            with open(rtf_path, 'rb') as rtf_file:
                rtf_content = rtf_file.read().decode('utf-8', errors='ignore')
            
            # Look for embedded base64 images
            # Pattern for base64 data in RTF files
            base64_pattern = r'\\pict[^{]*?{[^}]*?\\pngblip[^}]*?}|{\\pict[^}]*?\\pngblip[^}]*?}'
            hex_pattern = r'\\pict[^{]*?{[^}]*?\\(?:pngblip|jpegblip)[^}]*?([0-9A-Fa-f\s]+)}'
            
            # First try to find base64 encoded images
            base64_matches = re.findall(base64_pattern, rtf_content)
            
            # Process each match
            for i, match in enumerate(base64_matches):
                try:
                    # Extract the hexadecimal data, removing RTF control words
                    hex_data = re.sub(r'\\[a-z]+|\{|\}|\\|\s', '', match)
                    
                    if hex_data:
                        # Convert hex to binary
                        binary_data = bytes.fromhex(hex_data)
                        
                        # Try to determine the image type
                        img_ext = '.png'  # Default to PNG
                        if b'\xff\xd8\xff' in binary_data[:10]:  # JPEG signature
                            img_ext = '.jpg'
                        elif b'GIF8' in binary_data[:10]:  # GIF signature
                            img_ext = '.gif'
                        
                        # Save the image
                        img_filename = f"rtf_image_{i+1}{img_ext}"
                        img_path = os.path.join(output_dir, img_filename)
                        
                        with open(img_path, 'wb') as img_file:
                            img_file.write(binary_data)
                        
                        extracted_images.append(img_path)
                except Exception as e:
                    print(f"Error extracting image {i+1}: {str(e)}")
            
            # If no images found with the first method, try alternative patterns
            if not extracted_images:
                # Try to extract hexadecimal data directly
                hex_matches = re.findall(hex_pattern, rtf_content)
                
                for i, hex_data in enumerate(hex_matches):
                    try:
                        # Clean up the hex data
                        clean_hex = re.sub(r'\s', '', hex_data)
                        
                        if clean_hex:
                            # Convert hex to binary
                            binary_data = bytes.fromhex(clean_hex)
                            
                            # Try to determine the image type
                            img_ext = '.png'  # Default to PNG
                            if b'\xff\xd8\xff' in binary_data[:10]:  # JPEG signature
                                img_ext = '.jpg'
                            elif b'GIF8' in binary_data[:10]:  # GIF signature
                                img_ext = '.gif'
                            
                            # Save the image
                            img_filename = f"rtf_image_{i+1}{img_ext}"
                            img_path = os.path.join(output_dir, img_filename)
                            
                            with open(img_path, 'wb') as img_file:
                                img_file.write(binary_data)
                            
                            extracted_images.append(img_path)
                    except Exception as e:
                        print(f"Error extracting image {i+1} (hex method): {str(e)}")
            
            return extracted_images
        
        except Exception as e:
            print(f"Error extracting images from RTF: {str(e)}")
            return []
    
    def rtf_to_html(self, rtf_path: str, output_dir: str) -> Tuple[str, List[str]]:
        """
        Convert an RTF file to HTML format.
        
        Args:
            rtf_path: Path to the RTF file
            output_dir: Directory to save the HTML file and extracted images
            
        Returns:
            Tuple containing (html_path, list of extracted image paths)
        """
        try:
            # Create a unique HTML filename
            html_filename = f"rtf_preview_{uuid.uuid4().hex}.html"
            html_path = os.path.join(output_dir, html_filename)
            
            # Extract images from RTF
            extracted_images = self.extract_images_from_rtf(rtf_path, output_dir)
            
            # Read RTF content using a specialized RTF approach instead of mammoth
            try:
                # First try to read RTF using striprtf library if available
                try:
                    import striprtf.striprtf as striprtf
                    with open(rtf_path, 'rb') as rtf_file:
                        rtf_text = rtf_file.read().decode('utf-8', errors='ignore')
                        html_content = striprtf.rtf_to_html(rtf_text)
                except ImportError:
                    # Fallback to basic text extraction
                    with open(rtf_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text_content = f.read()
                        # Basic RTF to HTML conversion
                        # Remove RTF headers and control words
                        clean_text = re.sub(r'\\[a-z0-9]+', ' ', text_content)
                        clean_text = re.sub(r'\{|\}|\\', '', clean_text)
                        # Create simple HTML
                        html_content = f"<html><head><title>RTF Preview</title></head><body><pre>{html.escape(clean_text)}</pre></body></html>"
            except Exception as e:
                print(f"Error parsing RTF content: {str(e)}")
                # Absolute fallback
                html_content = f"<html><head><title>RTF Preview</title></head><body><div>Error processing RTF file: {html.escape(str(e))}</div></body></html>"
            
            # Enhance the HTML with CSS for better display
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Make sure we have head and body sections
            if not soup.head:
                head = soup.new_tag('head')
                soup.insert(0, head)
            if not soup.body:
                body = soup.new_tag('body')
                if soup.pre:
                    body.append(soup.pre.extract())
                soup.append(body)
            
            # Add CSS styling
            style_tag = soup.new_tag('style')
            style_tag.string = """
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                color: #333;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #2c3e50;
                margin-top: 24px;
                margin-bottom: 16px;
            }
            img {
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
                margin: 10px 0;
            }
            pre {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 4px;
                overflow-x: auto;
                white-space: pre-wrap;
            }
            """
            soup.head.append(style_tag)
            
            # Add extracted images to the HTML
            if extracted_images:
                image_container = soup.new_tag('div')
                image_container['class'] = 'extracted-images'
                image_container['style'] = 'margin-top: 30px; border-top: 1px solid #ddd; padding-top: 20px;'
                
                heading = soup.new_tag('h2')
                heading.string = "Embedded Images"
                image_container.append(heading)
                
                for i, img_path in enumerate(extracted_images):
                    img_tag = soup.new_tag('img')
                    img_tag['src'] = os.path.basename(img_path)
                    img_tag['alt'] = f"Embedded Image {i+1}"
                    img_tag['style'] = "display: block; margin: 20px 0; max-width: 100%;"
                    
                    figure = soup.new_tag('figure')
                    figure['style'] = "margin: 20px 0; text-align: center;"
                    
                    caption = soup.new_tag('figcaption')
                    caption.string = f"Embedded Image {i+1}"
                    caption['style'] = "font-style: italic; color: #666;"
                    
                    figure.append(img_tag)
                    figure.append(caption)
                    image_container.append(figure)
                
                # Add image container to body
                soup.body.append(image_container)
            
            # Write enhanced HTML to file
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(str(soup))
            
            return html_path, extracted_images
        
        except Exception as e:
            print(f"Error converting RTF to HTML: {str(e)}")
            import traceback
            print(f"Exception traceback: {traceback.format_exc()}")
            
            # Create a fallback HTML with error message
            fallback_html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; padding: 20px; }}
                    .error {{ color: red; background-color: #ffeeee; padding: 10px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>Error Converting RTF File</h1>
                <div class="error">
                    <p><strong>Error:</strong> {html.escape(str(e))}</p>
                    <p>The RTF file could not be properly converted to HTML.</p>
                </div>
                <hr>
                <h2>Raw Content Preview:</h2>
                <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow: auto;">
                """
            
            # Add the first 2000 characters of raw content
            try:
                with open(rtf_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_content = f.read(2000)
                    fallback_html += html.escape(raw_content) + "..."
            except:
                fallback_html += "Unable to read file content."
            
            fallback_html += """
                </pre>
            </body>
            </html>
            """
            
            # Write fallback HTML
            html_path = os.path.join(output_dir, f"rtf_preview_error_{uuid.uuid4().hex}.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(fallback_html)
            
            return html_path, [] 