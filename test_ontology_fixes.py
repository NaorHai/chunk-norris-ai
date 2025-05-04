import time
import logging
import traceback
from services.ai.ontology_graph_processor import OntologyGraphProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('app')

def test_with_different_inputs():
    """
    Test the ontology graph generation with different inputs
    """
    processor = OntologyGraphProcessor(host="127.0.0.1", port=6379)
    
    # Test case 1: Simple receipt with apostrophes
    logger.info("\n=== TEST CASE 1: Simple Receipt with Apostrophes ===")
    markdown_content1 = """
    # Harry's Bagels Receipt
    
    ## Store Information
    - Name: Harry's Bagels
    - Address: 520 8th Ave, New York, NY 10018
    - Phone: (646) 828-3371
    
    ## Order Details
    - Order #: 33
    - Date: 4/26/25
    - Time: 08:43:40 AM
    
    ## Items
    - Water: $2.38
    
    ## Payment
    - Total: $2.59
    - Card: American MasterCard ending in 1944
    - Auth: 817684
    """
    
    # Test case 2: Complex receipt with multiple items and special characters
    logger.info("\n=== TEST CASE 2: Complex Receipt ===")
    markdown_content2 = """
    # Joe's Coffee & Bagels
    
    ## Store Information
    - Name: Joe's Coffee & Bagels
    - Address: 123 Main St, Boston, MA 02108
    - Phone: (617) 555-1234
    
    ## Order Details
    - Order #: 42
    - Date: 4/27/25
    - Time: 09:15:30 AM
    
    ## Items
    - Coffee (Large): $3.50
    - Bagel with Cream Cheese: $4.25
    - Orange Juice: $2.99
    
    ## Payment
    - Subtotal: $10.74
    - Tax: $0.86
    - Total: $11.60
    - Card: Visa ending in 5678
    - Auth: 123456
    """
    
    # Test case 3: Minimal receipt
    logger.info("\n=== TEST CASE 3: Minimal Receipt ===")
    markdown_content3 = """
    # Quick Stop
    
    ## Order
    - Item: Soda
    - Price: $1.99
    
    ## Payment
    - Total: $1.99
    - Cash
    """
    
    test_cases = [
        ("Simple Receipt", markdown_content1),
        ("Complex Receipt", markdown_content2),
        ("Minimal Receipt", markdown_content3)
    ]
    
    for name, content in test_cases:
        logger.info(f"\nProcessing {name}...")
        start_time = time.time()
        
        try:
            # Generate ontology graph
            result = processor.generate_ontology_graph(content)
            
            # Verify the result structure
            assert "nodes" in result, f"Missing nodes in {name}"
            assert "edges" in result, f"Missing edges in {name}"
            
            # Verify node structure
            for node in result["nodes"]:
                assert "id" in node, f"Node missing id in {name}"
                assert "title" in node, f"Node missing title in {name}"
                assert "type" in node, f"Node missing type in {name}"
                assert "summary" in node, f"Node missing summary in {name}"
                assert "labels" in node, f"Node missing labels in {name}"
            
            # Verify edge structure
            for edge in result["edges"]:
                assert "source" in edge, f"Edge missing source in {name}"
                assert "target" in edge, f"Edge missing target in {name}"
                assert "relation" in edge, f"Edge missing relation in {name}"
                assert "properties" in edge, f"Edge missing properties in {name}"
            
            logger.info(f"✅ {name} processed successfully in {time.time() - start_time:.2f} seconds")
            logger.info(f"  - Nodes: {len(result['nodes'])}")
            logger.info(f"  - Edges: {len(result['edges'])}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == '__main__':
    test_with_different_inputs() 