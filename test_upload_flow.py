import logging
import json
from services.ai.ontology_graph_processor import OntologyGraphProcessor
from app import app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('app')

def test_upload_flow():
    """
    Test the complete upload flow including ontology graph generation
    """
    logger.info("\n=== TESTING UPLOAD FLOW ===")
    
    # Sample receipt data
    test_data = {
        "entities": [
            {
                "name": "Harry's Bagels",
                "description": "Store Information",
                "category": "Store",
                "properties": {
                    "address": "520 8th Ave, New York, NY 10018",
                    "phone": "(646) 828-3371"
                }
            },
            {
                "name": "Order #33",
                "description": "Order Details",
                "category": "Order",
                "properties": {
                    "receipt_number": "8533",
                    "date_time": "4/26/25 08:43:40 AM"
                }
            },
            {
                "name": "Water",
                "description": "Item Ordered",
                "category": "Item",
                "properties": {
                    "price": "$2.38"
                }
            },
            {
                "name": "Payment",
                "description": "Payment Information",
                "category": "Payment",
                "properties": {
                    "amount": "$2.59",
                    "card_type": "American MasterCard",
                    "last_four": "1944",
                    "auth_number": "817684"
                }
            }
        ],
        "relations": [
            {
                "source": "Harry's Bagels",
                "target": "Order #33",
                "type": "PROCESSED_ORDER",
                "description": "Store processed the order",
                "direction": "one-to-many"
            },
            {
                "source": "Order #33",
                "target": "Water",
                "type": "CONTAINS",
                "description": "Order contains item",
                "direction": "one-to-many"
            },
            {
                "source": "Order #33",
                "target": "Payment",
                "type": "PAID_BY",
                "description": "Order was paid by",
                "direction": "one-to-one"
            }
        ]
    }
    
    # Initialize processor
    processor = OntologyGraphProcessor(host="127.0.0.1", port=6379)
    
    # Step 1: Create graph queries
    logger.info("Step 1: Creating graph queries...")
    queries = processor.create_graph_queries(test_data["entities"], test_data["relations"])
    logger.info(f"Created {len(queries)} queries")
    
    # Step 2: Execute queries
    logger.info("Step 2: Executing queries...")
    graph_result = processor.execute_graph_queries(queries)
    logger.info(f"Graph result: {json.dumps(graph_result, indent=2)}")
    
    # Step 3: Format for UI
    logger.info("Step 3: Formatting for UI...")
    ui_formatted = processor._format_graph_for_ui(graph_result["nodes"], graph_result["edges"])
    logger.info(f"UI formatted: {json.dumps(ui_formatted, indent=2)}")
    
    # Step 4: Simulate upload response
    logger.info("Step 4: Simulating upload response...")
    upload_response = {
        "success": True,
        "filename": "test_receipt.pdf",
        "markdown": "Test markdown content",
        "used_ai_refinement": True,
        "used_chuck_norris_ai": True,
        "is_rtf": False,
        "memory_graph": None,
        "ontology_graph": ui_formatted
    }
    logger.info(f"Upload response: {json.dumps(upload_response, indent=2)}")
    
    # Verify the response structure
    logger.info("\nVerifying response structure:")
    logger.info(f"Response has ontology_graph: {'ontology_graph' in upload_response}")
    if 'ontology_graph' in upload_response:
        logger.info(f"Ontology graph has nodes: {'nodes' in upload_response['ontology_graph']}")
        logger.info(f"Ontology graph has edges: {'edges' in upload_response['ontology_graph']}")
        logger.info(f"Number of nodes: {len(upload_response['ontology_graph']['nodes'])}")
        logger.info(f"Number of edges: {len(upload_response['ontology_graph']['edges'])}")
    
    logger.info("=== END TESTING UPLOAD FLOW ===\n")

if __name__ == '__main__':
    test_upload_flow() 