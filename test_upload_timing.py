import time
import logging
from services.ai.ontology_graph_processor import OntologyGraphProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('app')

def test_upload_timing():
    """
    Test the timing and data flow of the ontology graph generation
    """
    logger.info("\n=== TESTING UPLOAD TIMING ===")
    
    # Sample markdown content
    markdown_content = """
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
    
    # Initialize processor
    processor = OntologyGraphProcessor(host="127.0.0.1", port=6379)
    
    # Step 1: Extract entities and relations
    logger.info("Step 1: Extracting entities and relations...")
    start_time = time.time()
    data = processor.extract_entities_and_relations(markdown_content)
    logger.info(f"Extraction time: {time.time() - start_time:.2f} seconds")
    logger.info(f"Found {len(data['entities'])} entities and {len(data['relations'])} relations")
    
    # Step 2: Create graph queries
    logger.info("\nStep 2: Creating graph queries...")
    start_time = time.time()
    queries = processor.create_graph_queries(data["entities"], data["relations"])
    logger.info(f"Query creation time: {time.time() - start_time:.2f} seconds")
    logger.info(f"Created {len(queries)} queries")
    
    # Step 3: Execute queries
    logger.info("\nStep 3: Executing queries...")
    start_time = time.time()
    graph_result = processor.execute_graph_queries(queries)
    logger.info(f"Query execution time: {time.time() - start_time:.2f} seconds")
    logger.info(f"Graph result: {graph_result}")
    
    # Step 4: Format for UI
    logger.info("\nStep 4: Formatting for UI...")
    start_time = time.time()
    ui_formatted = processor._format_graph_for_ui(graph_result["nodes"], graph_result["edges"])
    logger.info(f"UI formatting time: {time.time() - start_time:.2f} seconds")
    logger.info(f"UI formatted result: {ui_formatted}")
    
    # Total time
    logger.info("\n=== TIMING SUMMARY ===")
    logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
    logger.info(f"Final graph has {len(ui_formatted['nodes'])} nodes and {len(ui_formatted['edges'])} edges")

if __name__ == '__main__':
    test_upload_timing() 