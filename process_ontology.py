import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ai.ontology_graph_processor import OntologyGraphProcessor
import json

def main():
    # Read the markdown content
    with open('test_ontology.md', 'r') as f:
        markdown_content = f.read()
    
    # Initialize the processor
    processor = OntologyGraphProcessor()
    
    # Generate the ontology graph
    result = processor.generate_ontology_graph(markdown_content)
    
    # Print the result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main() 