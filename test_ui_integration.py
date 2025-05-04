import logging
from services.ai.ontology_graph_processor import OntologyGraphProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('app')

def test_ontology_format():
    processor = OntologyGraphProcessor(host="127.0.0.1", port=6379)
    
    # Receipt data
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
    
    # Create and execute queries
    queries = processor.create_graph_queries(test_data["entities"], test_data["relations"])
    result = processor.execute_graph_queries(queries)
    
    # Log results
    logger.info(f"Result type: {type(result)}")
    logger.info(f"Number of nodes: {len(result['nodes'])}")
    logger.info(f"Number of edges: {len(result['edges'])}")
    
    # Log the formatted nodes and edges
    logger.info("Formatted nodes:")
    for node in result['nodes']:
        logger.info(f"Node: {node}")
    
    logger.info("Formatted edges:")
    for edge in result['edges']:
        logger.info(f"Edge: {edge}")

if __name__ == '__main__':
    test_ontology_format() 