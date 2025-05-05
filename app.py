from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename
from flask_cors import CORS
import argparse
import time
import sys
import logging
import requests
import json
import concurrent.futures
from typing import Dict, Any, Tuple

# Import our new modular services
from services.factory import DocumentProcessorFactory
from services.ai import MemoryGraphProcessor, OntologyGraphProcessor
from logging_config import setup_logging
from config import OPENAI_API_KEY  # Import the API key from config

# Setup logging
logger = setup_logging()
if logger is None:
    print("Failed to setup logging")
    sys.exit(1)

# Configure Flask app
app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)  # Enable CORS for all routes

# Configure response timeout
app.config['RESPONSE_TIMEOUT'] = 180  # 3 minutes in seconds

# Configure upload folders
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['IMAGES_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
app.config['RTF_PREVIEWS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'rtf_previews')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size

# Create required directories
logger.info("Creating required directories...")
try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['IMAGES_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RTF_PREVIEWS_FOLDER'], exist_ok=True)
    logger.info(f"✅ Upload directory: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"✅ Images directory: {app.config['IMAGES_FOLDER']}")
    logger.info(f"✅ RTF previews directory: {app.config['RTF_PREVIEWS_FOLDER']}")
except Exception as e:
    logger.error(f"❌ Failed to create directories: {str(e)}")
    sys.exit(1)

@app.route('/test-logging')
def test_logging():
    """Test route to verify logging is working"""
    logger.info("TEST LOG: This is a test log message from the test route")
    return jsonify({"status": "success", "message": "Test log message sent"})

@app.route('/')
def index():
    logger.info("Home page accessed")
    return render_template('index.html')

@app.route('/uploads/images/<filename>')
def uploaded_image(filename):
    """Serve uploaded images"""
    logger.info(f"Serving image: {filename}")
    return send_from_directory(app.config['IMAGES_FOLDER'], filename)

def generate_document_summary(markdown_content: str) -> str:
    """
    Generate a document summary using GPT-4o Mini.
    
    Args:
        markdown_content: The markdown content to summarize
        
    Returns:
        A string containing the document summary
    """
    try:
        # Load the document summary template
        logger.info("🔹 Loading document summary template...")
        template_path = 'prompts/document_summary.mustache'
        if not os.path.exists(template_path):
            logger.error(f"❌ Document summary template not found at {template_path}")
            return None
            
        with open(template_path, 'r') as template_file:
            template_content = template_file.read()
            
        # Replace placeholder with actual content
        prompt = template_content.replace("{{content}}", markdown_content)
        
        # Call OpenAI API
        api_key = OPENAI_API_KEY
        if not api_key:
            logger.error("❌ OpenAI API key not found")
            return None
            
        api_url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2
        }
        
        logger.info("🔹 Sending request to OpenAI API...")
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            logger.error(f"❌ OpenAI API error: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
            
        result = response.json()
        
        if 'choices' not in result or not result['choices']:
            logger.error("❌ No choices in OpenAI API response")
            return None
            
        summary = result['choices'][0]['message']['content']
        logger.info(f"✅ Document summary generated successfully")
        logger.info(f"📝 Summary length: {len(summary)} characters")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Error generating document summary: {str(e)}")
        import traceback
        logger.error(f"❌ Exception traceback: {traceback.format_exc()}")
        return None

def process_document_summary(markdown_content: str) -> Tuple[str, float]:
    """Process document summary in parallel"""
    start_time = time.time()
    summary = generate_document_summary(markdown_content)
    end_time = time.time()
    return summary, end_time - start_time

def process_memory_graph(markdown_content: str) -> Tuple[Dict[str, Any], float]:
    """Process memory graph in parallel"""
    start_time = time.time()
    memory_graph_processor = MemoryGraphProcessor()
    memory_graph = memory_graph_processor.generate_memory_graph(markdown_content)
    end_time = time.time()
    return memory_graph, end_time - start_time

def process_ontology_graph(markdown_content: str) -> Tuple[Dict[str, Any], float]:
    """Process ontology graph in parallel"""
    start_time = time.time()
    ontology_graph_processor = OntologyGraphProcessor(host="127.0.0.1", port=6379)
    ontology_graph = ontology_graph_processor.generate_ontology_graph(markdown_content)
    end_time = time.time()
    return ontology_graph, end_time - start_time

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and document processing
    """
    # Log the start of the upload process
    logger.info("\n=================== DOCUMENT UPLOAD REQUEST ===================")
    logger.info(f"⏱️ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if file is provided
    if 'file' not in request.files:
        logger.error("❌ ERROR: No file part in request")
        logger.info("=================== END DOCUMENT UPLOAD REQUEST ===================\n")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("❌ ERROR: No file selected")
        logger.info("=================== END DOCUMENT UPLOAD REQUEST ===================\n")
        return jsonify({'error': 'No file selected'}), 400
    
    # Print request form data for debugging
    logger.info(f"📋 Request form data: {dict(request.form)}")
    logger.info(f"📄 File name: {file.filename}")
    file_size = len(file.read())
    logger.info(f"📊 File size: {file_size} bytes ({file_size/1024:.1f} KB)")
    file.seek(0)  # Reset file pointer after reading size
    
    # Get AI refinement preference (default to True if not specified)
    use_ai_refinement = request.form.get('use_ai_refinement', 'true').lower() == 'true'
    
    # Get Chuck Norris AI preference (default to False if not specified)
    use_chuck_norris_ai = request.form.get('use_chuck_norris_ai', 'false').lower() == 'true'
    
    # Get memory graph generation preference (default to False if not specified)
    generate_memory_graph = request.form.get('generate_memory_graph', 'false').lower() == 'true'
    
    # Get ontology graph generation preference (default to False if not specified)
    generate_ontology_graph = request.form.get('generate_ontology_graph', 'false').lower() == 'true'
    
    logger.info(f"⚙️ Processing options:")
    logger.info(f"  - AI refinement: {'✅' if use_ai_refinement else '❌'}")
    logger.info(f"  - Chuck Norris AI: {'✅' if use_chuck_norris_ai else '❌'}")
    logger.info(f"  - Memory graph: {'✅' if generate_memory_graph else '❌'}")
    logger.info(f"  - Ontology graph: {'✅' if generate_ontology_graph else '❌'}")
    
    start_time = time.time()
    
    try:
        # Generate a unique filename to avoid collisions
        original_filename = secure_filename(file.filename)
        file_ext = os.path.splitext(original_filename)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        logger.info(f"🔹 STEP 1: Saving uploaded file to {filepath}")
        file.save(filepath)
        logger.info(f"✅ File saved successfully")
        
        # Handle images differently - save them permanently for display
        is_image = file_ext.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        is_rtf = file_ext.lower() == '.rtf'
        is_pdf = file_ext.lower() == '.pdf'
        is_docx = file_ext.lower() in ['.doc', '.docx']
        
        file_type = 'image' if is_image else 'PDF' if is_pdf else 'Word document' if is_docx else 'RTF document' if is_rtf else 'document'
        logger.info(f"📄 File type detected: {file_type}")
        
        if is_image:
            # For images, save to images folder for persistence and reference
            image_path = os.path.join(app.config['IMAGES_FOLDER'], unique_filename)
            # If file was saved to temp location, move it
            if os.path.exists(filepath) and filepath != image_path:
                logger.info(f"🔹 Moving image to permanent storage: {image_path}")
                os.rename(filepath, image_path)
                filepath = image_path
                logger.info(f"✅ Image moved successfully")
        
        logger.info(f"\n🔹 STEP 2: Converting file '{original_filename}' to markdown")
        logger.info(f"📄 File path: {filepath}")
        
        # Use our factory to get the appropriate processor
        factory = DocumentProcessorFactory()
        memory_graph = None
        ontology_graph = None
        
        # Special handling for Chuck Norris AI mode
        if use_chuck_norris_ai:
            logger.info(f"🔹 Using Chuck Norris AI for processing")
            output_dir = app.config['RTF_PREVIEWS_FOLDER'] if is_rtf else None
            # Let the factory handle the processing based on file type
            result = factory.process_file(filepath, use_ai=True, output_dir=output_dir)
            
            # Check if result is a dictionary with markdown and html_preview
            if isinstance(result, dict) and 'markdown' in result:
                markdown_content = result['markdown']
                html_preview = result.get('html_preview')
                logger.info(f"✅ Markdown conversion successful using Chuck Norris AI")
                logger.info(f"📊 Markdown length: {len(markdown_content)} characters")
                if html_preview:
                    logger.info(f"📄 HTML preview available at: {html_preview}")
            else:
                # If not a dict, assume it's just markdown content
                markdown_content = result
                html_preview = None
                logger.info(f"✅ Markdown conversion successful using Chuck Norris AI")
                logger.info(f"📊 Markdown length: {len(markdown_content)} characters")
            
            # Generate memory graph if requested
            if generate_memory_graph:
                try:
                    # Initialize memory graph processor
                    logger.info(f"🔹 STEP 3: Generating memory graph")
                    memory_graph_processor = MemoryGraphProcessor()
                    memory_graph = memory_graph_processor.generate_memory_graph(markdown_content)
                    if memory_graph and 'nodes' in memory_graph:
                        logger.info(f"✅ Memory graph generation successful with {len(memory_graph['nodes'])} nodes and {len(memory_graph['edges'])} edges")
                    else:
                        logger.warning(f"⚠️ Memory graph generated but may be empty or invalid")
                except Exception as e:
                    logger.error(f"❌ Error generating memory graph: {str(e)}")
                    # Set memory_graph to None if there's an error
                    memory_graph = None
            
            # Generate document summary
            logger.info(f"🔹 STEP 4: Generating document summary")
            document_summary = generate_document_summary(markdown_content)
            if document_summary:
                logger.info(f"✅ Document summary generated successfully")
            else:
                logger.warning(f"⚠️ Document summary generation failed")
            
            # Generate ontology graph if requested
            if generate_ontology_graph:
                try:
                    # Initialize ontology graph processor
                    logger.info(f"🔹 STEP 5: Generating ontology graph")
                    ontology_graph_processor = OntologyGraphProcessor(host="127.0.0.1", port=6379)
                    ontology_graph = ontology_graph_processor.generate_ontology_graph(markdown_content)
                    if ontology_graph and 'nodes' in ontology_graph:
                        logger.info(f"✅ Ontology graph generation successful with {len(ontology_graph['nodes'])} nodes and {len(ontology_graph['edges'])} edges")
                    else:
                        logger.warning(f"⚠️ Ontology graph generated but may be empty or invalid")
                except Exception as e:
                    logger.error(f"❌ Error generating ontology graph: {str(e)}")
                    # Set ontology_graph to None if there's an error
                    ontology_graph = None
            
            # Calculate and log processing time
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"🎉 Document processing completed successfully:")
            logger.info(f"  - Original file: {original_filename}")
            logger.info(f"  - Markdown length: {len(markdown_content)} characters")
            logger.info(f"  - Processing time: {total_time:.2f} seconds")
            
            if generate_memory_graph and memory_graph:
                node_count = len(memory_graph.get('nodes', []))
                edge_count = len(memory_graph.get('edges', []))
                logger.info(f"  - Memory graph: {node_count} nodes, {edge_count} edges")
            
            if generate_ontology_graph and ontology_graph:
                node_count = len(ontology_graph.get('nodes', []))
                edge_count = len(ontology_graph.get('edges', []))
                logger.info(f"  - Ontology graph: {node_count} nodes, {edge_count} edges")
            
            logger.info("=================== END DOCUMENT UPLOAD REQUEST ===================\n")
            
            # For Chuck Norris AI files, return appropriate response
            return jsonify({
                'success': True,
                'filename': original_filename,
                'markdown': markdown_content,
                'html_preview': html_preview,
                'used_ai_refinement': use_ai_refinement,
                'used_chuck_norris_ai': use_chuck_norris_ai,
                'is_rtf': is_rtf,
                'memory_graph': memory_graph,
                'ontology_graph': ontology_graph,
                'document_summary': document_summary,
                'processing_time': round(total_time, 2)
            })
        
        # Standard processing flow for non-Chuck Norris AI mode
        logger.info(f"🔹 Using standard processing")
        output_dir = app.config['RTF_PREVIEWS_FOLDER'] if is_rtf else None
        result = factory.process_file(filepath, use_ai=False, output_dir=output_dir)
        
        # Check if result is a dictionary with markdown and html_preview
        if isinstance(result, dict) and 'markdown' in result:
            markdown_content = result['markdown']
            html_preview = result.get('html_preview')
            logger.info(f"✅ Markdown conversion successful with standard processing")
            logger.info(f"📊 Markdown length: {len(markdown_content)} characters")
            if html_preview:
                logger.info(f"📄 HTML preview available at: {html_preview}")
        else:
            # If not a dict, assume it's just markdown content
            markdown_content = result
            html_preview = None
            logger.info(f"✅ Markdown conversion successful with standard processing")
            logger.info(f"📊 Markdown length: {len(markdown_content)} characters")
        
        # Generate memory graph if requested
        if generate_memory_graph:
            try:
                # Initialize memory graph processor
                logger.info(f"🔹 STEP 3: Generating memory graph")
                memory_graph_processor = MemoryGraphProcessor()
                memory_graph = memory_graph_processor.generate_memory_graph(markdown_content)
                if memory_graph and 'nodes' in memory_graph:
                    logger.info(f"✅ Memory graph generation successful with {len(memory_graph['nodes'])} nodes and {len(memory_graph['edges'])} edges")
                else:
                    logger.warning(f"⚠️ Memory graph generated but may be empty or invalid")
            except Exception as e:
                logger.error(f"❌ Error generating memory graph: {str(e)}")
                # Set memory_graph to None if there's an error
                memory_graph = None
        
        # Generate document summary
        logger.info(f"🔹 STEP 4: Generating document summary")
        document_summary = generate_document_summary(markdown_content)
        if document_summary:
            logger.info(f"✅ Document summary generated successfully")
        else:
            logger.warning(f"⚠️ Document summary generation failed")
        
        # Generate ontology graph if requested
        if generate_ontology_graph:
            try:
                # Initialize ontology graph processor
                logger.info(f"🔹 STEP 5: Generating ontology graph")
                ontology_graph_processor = OntologyGraphProcessor(host="127.0.0.1", port=6379)
                ontology_graph = ontology_graph_processor.generate_ontology_graph(markdown_content)
                if ontology_graph and 'nodes' in ontology_graph:
                    logger.info(f"✅ Ontology graph generation successful with {len(ontology_graph['nodes'])} nodes and {len(ontology_graph['edges'])} edges")
                else:
                    logger.warning(f"⚠️ Ontology graph generated but may be empty or invalid")
            except Exception as e:
                logger.error(f"❌ Error generating ontology graph: {str(e)}")
                # Set ontology_graph to None if there's an error
                ontology_graph = None
        
        # For images, update the markdown content to point to the correct URL
        if is_image:
            # Keep the image file for display
            logger.info(f"\n🎉 Image processing completed successfully:")
            logger.info(f"  - Original file: {original_filename}")
            logger.info(f"  - Markdown length: {len(markdown_content)} characters")
            logger.info(f"  - Image path: /uploads/images/{unique_filename}")
            
            # Calculate and log processing time
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"  - Processing time: {total_time:.2f} seconds")
            
            if generate_memory_graph and memory_graph:
                node_count = len(memory_graph.get('nodes', []))
                edge_count = len(memory_graph.get('edges', []))
                logger.info(f"  - Memory graph: {node_count} nodes, {edge_count} edges")
            
            if generate_ontology_graph and ontology_graph:
                node_count = len(ontology_graph.get('nodes', []))
                edge_count = len(ontology_graph.get('edges', []))
                logger.info(f"  - Ontology graph: {node_count} nodes, {edge_count} edges")
            
            logger.info("=================== END DOCUMENT UPLOAD REQUEST ===================\n")
            
            return jsonify({
                'success': True,
                'filename': original_filename,
                'markdown': markdown_content,
                'imagePath': f"/uploads/images/{unique_filename}",
                'used_ai_refinement': use_ai_refinement,
                'used_chuck_norris_ai': use_chuck_norris_ai,
                'is_rtf': False,
                'memory_graph': memory_graph,
                'ontology_graph': ontology_graph,
                'document_summary': document_summary,
                'processing_time': round(total_time, 2)
            })
        else:
            # For non-images that aren't RTF with html_preview, clean up after conversion
            if os.path.exists(filepath) and not is_image and not html_preview:
                logger.info(f"🧹 Cleaning up temporary file: {filepath}")
                os.remove(filepath)
                logger.info(f"✅ Temporary file removed")
            
            # Calculate and log processing time
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"🎉 Document processing completed successfully:")
            logger.info(f"  - Original file: {original_filename}")
            logger.info(f"  - Markdown length: {len(markdown_content)} characters")
            logger.info(f"  - Processing time: {total_time:.2f} seconds")
            
            if generate_memory_graph and memory_graph:
                node_count = len(memory_graph.get('nodes', []))
                edge_count = len(memory_graph.get('edges', []))
                logger.info(f"  - Memory graph: {node_count} nodes, {edge_count} edges")
            
            if generate_ontology_graph and ontology_graph:
                node_count = len(ontology_graph.get('nodes', []))
                edge_count = len(ontology_graph.get('edges', []))
                logger.info(f"  - Ontology graph: {node_count} nodes, {edge_count} edges")
            
            logger.info("=================== END DOCUMENT UPLOAD REQUEST ===================\n")
            
            return jsonify({
                'success': True,
                'filename': original_filename,
                'markdown': markdown_content,
                'html_preview': html_preview,
                'used_ai_refinement': use_ai_refinement,
                'used_chuck_norris_ai': use_chuck_norris_ai,
                'is_rtf': is_rtf,
                'memory_graph': memory_graph,
                'ontology_graph': ontology_graph,
                'document_summary': document_summary,
                'processing_time': round(total_time, 2)
            })
    except Exception as e:
        logger.error(f"❌ ERROR processing file: {str(e)}")
        import traceback
        logger.error(f"❌ Exception traceback: {traceback.format_exc()}")
        # Clean up the uploaded file in case of error
        if os.path.exists(filepath):
            logger.info(f"🧹 Cleaning up file after error: {filepath}")
            os.remove(filepath)
            logger.info(f"✅ File removed")
        logger.info("=================== END DOCUMENT UPLOAD REQUEST (WITH ERROR) ===================\n")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_memory_graph', methods=['POST'])
def generate_memory_graph():
    """
    Generate a memory graph from markdown content
    """
    try:
        data = request.get_json()
        if not data or 'markdown' not in data:
            return jsonify({'error': 'No markdown content provided'}), 400
        
        markdown_content = data['markdown']
        
        logger.info("\n=================== MEMORY GRAPH REQUEST ===================")
        logger.info(f"⏱️ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Received markdown content of length: {len(markdown_content)} characters")
        
        start_time = time.time()
        
        # Initialize the memory graph processor
        logger.info(f"🔹 Initializing MemoryGraphProcessor")
        memory_graph_processor = MemoryGraphProcessor()
        
        # Generate the memory graph
        logger.info(f"🔹 Generating memory graph from markdown content")
        memory_graph = memory_graph_processor.generate_memory_graph(markdown_content)
        
        # Log the results
        end_time = time.time()
        total_time = end_time - start_time
        node_count = len(memory_graph.get('nodes', []))
        edge_count = len(memory_graph.get('edges', []))
        
        logger.info(f"✅ Memory graph generation completed in {total_time:.2f} seconds")
        logger.info(f"📊 Memory graph statistics:")
        logger.info(f"  - Nodes: {node_count}")
        logger.info(f"  - Edges: {edge_count}")
        logger.info(f"  - Node-to-edge ratio: {node_count/edge_count:.2f}" if edge_count > 0 else "  - No edges found")
        
        if node_count == 0:
            logger.warning(f"⚠️ WARNING: The generated memory graph contains no nodes")
        elif edge_count == 0:
            logger.warning(f"⚠️ WARNING: The generated memory graph contains no edges between nodes")
        else:
            logger.info(f"🎉 Memory graph successfully generated with {node_count} nodes and {edge_count} edges")
        
        logger.info("=================== END MEMORY GRAPH REQUEST ===================\n")
        
        return jsonify({
            'success': True,
            'memory_graph': memory_graph
        })
    except Exception as e:
        logger.error(f"❌ ERROR generating memory graph: {str(e)}")
        import traceback
        logger.error(f"❌ Exception traceback: {traceback.format_exc()}")
        logger.info("=================== END MEMORY GRAPH REQUEST (WITH ERROR) ===================\n")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_ontology_graph', methods=['POST'])
def generate_ontology_graph():
    """
    Generate an ontology graph from markdown content using FalkorDB's GraphRAG-SDK
    """
    try:
        data = request.get_json()
        if not data or 'markdown' not in data:
            return jsonify({'error': 'No markdown content provided'}), 400
        
        markdown_content = data['markdown']
        
        logger.info("\n=================== ONTOLOGY GRAPH REQUEST ===================")
        logger.info(f"⏱️ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Received markdown content of length: {len(markdown_content)} characters")
        
        start_time = time.time()
        
        # Initialize the ontology graph processor
        logger.info(f"🔹 Initializing OntologyGraphProcessor")
        logger.info(f"🔸 Connecting to FalkorDB at 127.0.0.1:6379")
        ontology_graph_processor = OntologyGraphProcessor(host="127.0.0.1", port=6379)
        
        # Generate the ontology graph
        logger.info(f"🔹 Generating ontology graph from markdown content")
        ontology_graph = ontology_graph_processor.generate_ontology_graph(markdown_content)
        
        # Log the results
        end_time = time.time()
        total_time = end_time - start_time
        node_count = len(ontology_graph.get('nodes', []))
        edge_count = len(ontology_graph.get('edges', []))
        
        logger.info(f"✅ Ontology graph generation completed in {total_time:.2f} seconds")
        logger.info(f"📊 Ontology graph statistics:")
        logger.info(f"  - Nodes: {node_count}")
        logger.info(f"  - Edges: {edge_count}")
        logger.info(f"  - Node-to-edge ratio: {node_count/edge_count:.2f}" if edge_count > 0 else "  - No edges found")
        
        if node_count == 0:
            logger.warning(f"⚠️ WARNING: The generated ontology graph contains no nodes")
        elif edge_count == 0:
            logger.warning(f"⚠️ WARNING: The generated ontology graph contains no edges between nodes")
        else:
            # Count entity types if available
            entity_nodes = [n for n in ontology_graph.get('nodes', []) if n.get('type') == 'Entity']
            property_nodes = [n for n in ontology_graph.get('nodes', []) if n.get('type') == 'Property']
            logger.info(f"  - Entity nodes: {len(entity_nodes)}")
            logger.info(f"  - Property nodes: {len(property_nodes)}")
            logger.info(f"🎉 Ontology graph successfully generated with {node_count} nodes and {edge_count} edges")
        
        logger.info("=================== END ONTOLOGY GRAPH REQUEST ===================\n")
        
        return jsonify({
            'success': True,
            'ontology_graph': ontology_graph
        })
    except Exception as e:
        logger.error(f"❌ ERROR generating ontology graph: {str(e)}")
        import traceback
        logger.error(f"❌ Exception traceback: {traceback.format_exc()}")
        logger.info("=================== END ONTOLOGY GRAPH REQUEST (WITH ERROR) ===================\n")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Chuck Norris AI application')
    parser.add_argument('--port', type=int, default=8080, help='Port number to run the application on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the application on')
    parser.add_argument('--debug', action='store_true', help='Run the application in debug mode')
    parser.add_argument('--timeout', type=int, default=180, help='Response timeout in seconds')
    args = parser.parse_args()
    
    # Print configurations
    logger.info("\n=== CHUCK NORRIS AI STARTUP ===")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Response timeout: {args.timeout} seconds")
    
    # Apply timeout configuration
    app.config['RESPONSE_TIMEOUT'] = args.timeout
    
    # Test FalkorDB connection
    try:
        import redis
        falkordb_host = "127.0.0.1"
        falkordb_port = 6379  # Default FalkorDB port
        
        logger.info(f"\n=== TESTING FALKORDB CONNECTION ===")
        logger.info(f"Attempting to connect to FalkorDB at {falkordb_host}:{falkordb_port}")
        
        redis_client = redis.Redis(host=falkordb_host, port=falkordb_port)
        ping_response = redis_client.ping()
        
        if ping_response:
            logger.info(f"✅ Successfully connected to FalkorDB at {falkordb_host}:{falkordb_port}")
            
            # Check if Redis Graph module is loaded
            try:
                modules = redis_client.execute_command('MODULE LIST')
                graph_module_found = False
                
                for module in modules:
                    if b'graph' in str(module).lower():
                        graph_module_found = True
                        logger.info(f"✅ Redis Graph module is loaded")
                        break
                
                if not graph_module_found:
                    logger.warning(f"⚠️ Redis Graph module is not loaded in FalkorDB. Graph functionality may not work.")
            except Exception as e:
                logger.warning(f"⚠️ Could not check Redis Graph module: {str(e)}")
        else:
            logger.error(f"❌ Failed to connect to FalkorDB at {falkordb_host}:{falkordb_port}")
    except Exception as e:
        logger.error(f"❌ Error connecting to FalkorDB: {str(e)}")
    
    logger.info("=== END FALKORDB CONNECTION TEST ===\n")
    logger.info("Starting application...\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug) 