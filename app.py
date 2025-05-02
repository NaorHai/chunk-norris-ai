from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename
from file_parsers import convert_to_markdown, refine_markdown_with_llm
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['IMAGES_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size

# Create uploads directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGES_FOLDER'], exist_ok=True)

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
    
    # Get AI refinement preference (default to True if not specified)
    use_ai_refinement = request.form.get('use_ai_refinement', 'true').lower() == 'true'
    
    # Get Chuck Norris AI preference (default to False if not specified)
    use_chuck_norris_ai = request.form.get('use_chuck_norris_ai', 'false').lower() == 'true'
    
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
            
            if is_image:
                # For images, save to images folder for persistence and reference
                image_path = os.path.join(app.config['IMAGES_FOLDER'], unique_filename)
                # If file was saved to temp location, move it
                if os.path.exists(filepath) and filepath != image_path:
                    os.rename(filepath, image_path)
                    filepath = image_path
            
            # First convert the file to raw markdown
            raw_markdown = convert_to_markdown(filepath, use_ai_refinement=False, use_chuck_norris_ai=use_chuck_norris_ai)
            
            # Refine with LLM if requested (and Chuck Norris AI wasn't used)
            if use_ai_refinement and not use_chuck_norris_ai:
                markdown_content = refine_markdown_with_llm(raw_markdown)
            else:
                markdown_content = raw_markdown
            
            # For images, update the markdown content to point to the correct URL
            if is_image:
                # Keep the image file for display
                return jsonify({
                    'success': True,
                    'filename': original_filename,
                    'markdown': markdown_content,
                    'imagePath': f"/uploads/images/{unique_filename}",
                    'used_ai_refinement': use_ai_refinement,
                    'used_chuck_norris_ai': use_chuck_norris_ai
                })
            else:
                # For non-images, clean up after conversion
                if os.path.exists(filepath) and not is_image:
                    os.remove(filepath)
                
                return jsonify({
                    'success': True,
                    'filename': original_filename,
                    'markdown': markdown_content,
                    'used_ai_refinement': use_ai_refinement,
                    'used_chuck_norris_ai': use_chuck_norris_ai
                })
        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to allow external access and port 8090 to avoid conflicts
    app.run(debug=True, host='0.0.0.0', port=8090) 