from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Import our new modular services
from services.factory import DocumentProcessorFactory

app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['IMAGES_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
app.config['RTF_PREVIEWS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'rtf_previews')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size

# Create uploads directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGES_FOLDER'], exist_ok=True)
os.makedirs(app.config['RTF_PREVIEWS_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/images/<filename>')
def uploaded_image(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['IMAGES_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Print request form data for debugging
    print("\n=== UPLOAD REQUEST DATA ===")
    print(f"Form data: {request.form}")
    
    # Get AI refinement preference (default to True if not specified)
    use_ai_refinement = request.form.get('use_ai_refinement', 'true').lower() == 'true'
    
    # Get Chuck Norris AI preference (default to False if not specified)
    use_chuck_norris_ai = request.form.get('use_chuck_norris_ai', 'false').lower() == 'true'
    
    print(f"use_ai_refinement: {use_ai_refinement}")
    print(f"use_chuck_norris_ai: {use_chuck_norris_ai}")
    print("=== END REQUEST DATA ===\n")
    
    if file:
        # Generate a unique filename to avoid collisions
        original_filename = secure_filename(file.filename)
        file_ext = os.path.splitext(original_filename)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)
        
        try:
            # Handle images differently - save them permanently for display
            is_image = file_ext.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
            is_rtf = file_ext.lower() == '.rtf'
            is_pdf = file_ext.lower() == '.pdf'
            is_docx = file_ext.lower() in ['.doc', '.docx']
            
            if is_image:
                # For images, save to images folder for persistence and reference
                image_path = os.path.join(app.config['IMAGES_FOLDER'], unique_filename)
                # If file was saved to temp location, move it
                if os.path.exists(filepath) and filepath != image_path:
                    os.rename(filepath, image_path)
                    filepath = image_path
            
            print(f"Converting file '{original_filename}' to markdown...")
            print(f"File type: {'image' if is_image else 'pdf' if is_pdf else 'docx' if is_docx else 'rtf' if is_rtf else 'document'}")
            print(f"File path: {filepath}")
            
            # Use our factory to get the appropriate processor
            factory = DocumentProcessorFactory()
            
            # Special handling for Chuck Norris AI mode
            if use_chuck_norris_ai:
                output_dir = app.config['RTF_PREVIEWS_FOLDER'] if is_rtf else None
                # Let the factory handle the processing based on file type
                result = factory.process_file(filepath, use_ai=True, output_dir=output_dir)
                
                # Check if result is a dictionary with markdown and html_preview
                if isinstance(result, dict) and 'markdown' in result:
                    markdown_content = result['markdown']
                    html_preview = result.get('html_preview')
                else:
                    # If not a dict, assume it's just markdown content
                    markdown_content = result
                    html_preview = None
                
                # For Chuck Norris AI files, return appropriate response
                return jsonify({
                    'success': True,
                    'filename': original_filename,
                    'markdown': markdown_content,
                    'html_preview': html_preview,
                    'used_ai_refinement': use_ai_refinement,
                    'used_chuck_norris_ai': use_chuck_norris_ai,
                    'is_rtf': is_rtf
                })
            
            # Standard processing flow for non-Chuck Norris AI mode
            output_dir = app.config['RTF_PREVIEWS_FOLDER'] if is_rtf else None
            result = factory.process_file(filepath, use_ai=False, output_dir=output_dir)
            
            # Check if result is a dictionary with markdown and html_preview
            if isinstance(result, dict) and 'markdown' in result:
                markdown_content = result['markdown']
                html_preview = result.get('html_preview')
            else:
                # If not a dict, assume it's just markdown content
                markdown_content = result
                html_preview = None
            
            # For images, update the markdown content to point to the correct URL
            if is_image:
                # Keep the image file for display
                return jsonify({
                    'success': True,
                    'filename': original_filename,
                    'markdown': markdown_content,
                    'imagePath': f"/uploads/images/{unique_filename}",
                    'used_ai_refinement': use_ai_refinement,
                    'used_chuck_norris_ai': use_chuck_norris_ai,
                    'is_rtf': False
                })
            else:
                # For non-images that aren't RTF with html_preview, clean up after conversion
                if os.path.exists(filepath) and not is_image and not html_preview:
                    os.remove(filepath)
                
                return jsonify({
                    'success': True,
                    'filename': original_filename,
                    'markdown': markdown_content,
                    'html_preview': html_preview,
                    'used_ai_refinement': use_ai_refinement,
                    'used_chuck_norris_ai': use_chuck_norris_ai,
                    'is_rtf': is_rtf
                })
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            import traceback
            print(f"Exception traceback: {traceback.format_exc()}")
            # Clean up the uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to allow external access and port 8081 to avoid conflicts
    app.run(debug=True, host='0.0.0.0', port=8081) 