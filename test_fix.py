import logging
from services.ai.ontology_graph_processor import OntologyGraphProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('app')

def test_node_names():
    processor = OntologyGraphProcessor(host="127.0.0.1", port=6379)
    
    # Test data with explicit names
    test_data = {
        "entities": [
            {
                "name": "Flight",
                "description": "Details of the flight including departure and arrival information.",
                "category": "General"
            },
            {
                "name": "Passenger",
                "description": "Information about the passenger traveling on the flight.",
                "category": "General"
            }
        ],
        "relations": [
            {
                "source": "Passenger",
                "target": "Flight",
                "type": "books",
                "description": "The passenger is booked on the flight.",
                "direction": "Passenger to Flight"
            }
        ]
    }
    
    # Create and execute queries
    queries = processor.create_graph_queries(test_data["entities"], test_data["relations"])
    result = processor.execute_graph_queries(queries)
    
    # Log results
    logger.info(f"Result: {result}")
    logger.info(f"Number of nodes: {len(result['nodes'])}")
    logger.info(f"Number of edges: {len(result['edges'])}")
    
    # Print node details
    for node in result['nodes']:
        logger.info(f"Node: {node}")
    
    # Print edge details
    for edge in result['edges']:
        logger.info(f"Edge: {edge}")

if __name__ == '__main__':
    test_node_names() 