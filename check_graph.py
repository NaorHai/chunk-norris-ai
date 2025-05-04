from falkordb import FalkorDB
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('app')

def check_graph_state():
    try:
        # Connect to FalkorDB
        graph = FalkorDB(host="127.0.0.1", port=6379)
        graph_db = graph.select_graph('ontology_graph')
        
        # Query nodes
        nodes_result = graph_db.query("MATCH (n) RETURN n, labels(n) as labels, properties(n) as props")
        logger.info(f"Found {len(nodes_result.result_set)} nodes")
        
        # Print node details
        for i, record in enumerate(nodes_result.result_set):
            logger.info(f"Node {i}:")
            logger.info(f"  Labels: {record[1]}")
            logger.info(f"  Properties: {record[2]}")
        
        # Query edges
        edges_result = graph_db.query("""
            MATCH (source)-[r]->(target)
            RETURN type(r) as type,
                   properties(r) as props,
                   source.name as source,
                   target.name as target
        """)
        logger.info(f"Found {len(edges_result.result_set)} edges")
        
        # Print edge details
        for i, record in enumerate(edges_result.result_set):
            logger.info(f"Edge {i}:")
            logger.info(f"  Type: {record[0]}")
            logger.info(f"  Properties: {record[1]}")
            logger.info(f"  Source: {record[2]}")
            logger.info(f"  Target: {record[3]}")
            
    except Exception as e:
        logger.error(f"Error checking graph state: {str(e)}")

if __name__ == '__main__':
    check_graph_state() 