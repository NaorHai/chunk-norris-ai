import os
import tempfile
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class HtmlRenderer:
    """
    Renders HTML content to images for AI processing
    """
    
    @staticmethod
    def html_to_image(html_path: str, output_path: str) -> Optional[str]:
        """
        Convert HTML file to an image for LLM processing.
        
        Args:
            html_path: Path to the HTML file
            output_path: Path to save the output image
            
        Returns:
            Path to the generated image or None if failed
        """
        try:
            # This requires wkhtmltoimage to be installed on the system
            # Try to use Python wkhtmltopdf library if available
            try:
                import imgkit
                options = {
                    'width': 1200,
                    'height': 1600,
                    'quality': 80,
                    'enable-local-file-access': True
                }
                imgkit.from_file(html_path, output_path, options=options)
                return output_path
            except ImportError:
                print("imgkit not installed, trying alternative method...")
                
                # Alternative approach using wkhtmltoimage command line
                import subprocess
                try:
                    # Try to use wkhtmltoimage directly
                    command = [
                        'wkhtmltoimage',
                        '--width', '1200',
                        '--height', '1600',
                        '--quality', '80',
                        '--enable-local-file-access',
                        html_path,
                        output_path
                    ]
                    subprocess.run(command, check=True)
                    return output_path
                except (subprocess.SubprocessError, FileNotFoundError):
                    print("wkhtmltoimage command failed, trying Python rendering...")
                    
                    # Fallback to a very basic HTML rendering with PIL
                    if PIL_AVAILABLE:
                        text = "HTML Preview (rendering failed)"
                        img = Image.new('RGB', (800, 200), color=(255, 255, 255))
                        try:
                            draw = ImageDraw.Draw(img)
                            try:
                                font = ImageFont.truetype("Arial", 20)
                            except:
                                font = ImageFont.load_default()
                            draw.text((20, 20), text, fill=(0, 0, 0), font=font)
                        except:
                            pass  # Can't even draw text, will return blank image
                        
                        img.save(output_path)
                        return output_path
                    else:
                        print("PIL not available, cannot create fallback image")
                        return None
        except Exception as e:
            print(f"Error converting HTML to image: {str(e)}")
            return None 