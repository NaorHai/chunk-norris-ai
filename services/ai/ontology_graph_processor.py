import os
import json
import logging
import traceback
from typing import Dict, Any, List
import time
from openai import OpenAI
from falkordb import FalkorDB
from config import OPENAI_API_KEY
import pystache

# Get logger from logging_config
logger = logging.getLogger('app')

class OntologyGraphProcessor:
    """
    Processor that generates an ontology graph from markdown content using OpenAI GPT-4
    and stores it in FalkorDB.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379):
        """
        Initialize the ontology graph processor
        
        Args:
            host: FalkorDB host (defaults to localhost)
            port: FalkorDB port (defaults to 6379)
        """
        self.host = host
        self.port = port
        self.graph = None
        self.graph_db = None
        self.logger = logger
        
        self.connect()
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Load the prompt template
        self.prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        try:
            template_path = os.path.join('prompts', 'ontology_graph_prompt.mustache')
            self.logger.info(f"🔹 Loading prompt template from: {template_path}")
            
            if not os.path.exists(template_path):
                self.logger.error(f"❌ Template file not found at: {template_path}")
                raise FileNotFoundError(f"Template file not found at: {template_path}")
                
            with open(template_path, 'r') as f:
                template = f.read()
                self.logger.info(f"🔹 Template content length: {len(template)} characters")
                self.logger.info(f"🔹 Template content preview: {template[:200]}...")
                return template
        except Exception as e:
            self.logger.error(f"❌ Error loading prompt template: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
    def connect(self):
        """Connect to FalkorDB and initialize the graph database."""
        try:
            self.graph = FalkorDB(host=self.host, port=self.port)
            self.graph_db = self.graph.select_graph('ontology_graph')
            self.logger.info(f"🔹 Initializing OntologyGraphProcessor with host={self.host} and port={self.port}")
        except Exception as e:
            self.logger.error(f"❌ Error connecting to FalkorDB: {str(e)}")
            raise
        
    def extract_entities_and_relations(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relations from the given text using GPT-4.
        This method is designed to be generic and handle various types of content.
        """
        logger.info("🔹 Extracting entities and relations...")
        
        try:
            # Render the prompt template
            rendered_prompt = pystache.render(self.prompt_template, {
                'markdown_content': text
            })
            
            # Log the rendered prompt for debugging
            logger.info(f"🔹 Rendered prompt length: {len(rendered_prompt)} characters")
            logger.info(f"🔹 Rendered prompt preview: {rendered_prompt[:200]}...")
            
            # Split the rendered prompt into system and user messages
            parts = rendered_prompt.split('---USER_PROMPT_START---')
            if len(parts) != 2:
                logger.error(f"❌ Failed to split prompt template correctly. Got {len(parts)} parts instead of 2")
                logger.error(f"Template content: {rendered_prompt}")
                return {"entities": [], "relations": []}
                
            system_message = parts[0].replace('{{! System prompt for ontology graph generation }}', '').strip()
            user_message = parts[1].split('---USER_PROMPT_END---')[0].strip()
            
            # Log the split messages for debugging
            logger.info(f"🔹 System message length: {len(system_message)} characters")
            logger.info(f"🔹 User message length: {len(user_message)} characters")
            
            # Call GPT-4
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0
            )
            
            # Extract the response text
            result_text = response.choices[0].message.content
            
            # Parse the JSON response
            try:
                result = json.loads(result_text)
                
                # Validate and clean up the result
                if not isinstance(result, dict):
                    raise ValueError("Result is not a dictionary")
                
                if "entities" not in result:
                    result["entities"] = []
                if "relations" not in result:
                    result["relations"] = []
                
                # Ensure all entities have required fields
                for entity in result["entities"]:
                    if "label" not in entity:
                        entity["label"] = "Entity"
                    if "attributes" not in entity:
                        entity["attributes"] = {}
                    if "name" not in entity["attributes"]:
                        entity["attributes"]["name"] = entity.get("label", "Unknown")
                    if "category" not in entity["attributes"]:
                        entity["attributes"]["category"] = "General"
                    if "description" not in entity["attributes"]:
                        entity["attributes"]["description"] = entity.get("description", "")
                
                # Ensure all relations have required fields
                for relation in result["relations"]:
                    if "label" not in relation:
                        relation["label"] = "Related"
                    if "attributes" not in relation:
                        relation["attributes"] = {}
                    if "direction" not in relation["attributes"]:
                        relation["attributes"]["direction"] = "unidirectional"
                    if "type" not in relation["attributes"]:
                        relation["attributes"]["type"] = "general"
                    if "description" not in relation["attributes"]:
                        relation["attributes"]["description"] = relation.get("description", "")
                
                logger.info("✅ Successfully extracted entities and relations")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse JSON response: {str(e)}")
                logger.error(f"Response text: {result_text}")
                return {"entities": [], "relations": []}
                
        except Exception as e:
            logger.error(f"❌ Error in entity extraction: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"entities": [], "relations": []}
            
    def create_graph_queries(self, entities, relations):
        """
        Create graph queries for entities and relations.
        """
        queries = []
        
        # Create node queries
        for entity in entities:
            query = """
                CREATE (:Entity {
                    name: $name,
                    description: $description,
                    category: $category
                })
            """
            params = {
                "name": entity["name"],
                "description": entity["description"],
                "category": entity["category"]
            }
            queries.append((query, params))
        
        # Create edge queries
        for relation in relations:
            # Clean up relationship type (replace spaces with underscores)
            rel_type = relation["type"].replace(" ", "_")
            
            # Escape apostrophes in the description
            description = relation["description"].replace("'", "\\'")
            query = f"""
                MATCH (source:Entity {{name: $source_name}})
                MATCH (target:Entity {{name: $target_name}})
                CREATE (source)-[:{rel_type} {{
                    description: '{description}',
                    type: '{relation["type"]}',
                    direction: '{relation["direction"]}'
                }}]->(target)
            """
            params = {
                "source_name": relation["source"],
                "target_name": relation["target"]
            }
            queries.append((query, params))
        
        return queries

    def execute_graph_queries(self, queries: List[str]) -> Dict[str, Any]:
        """
        Execute the queries in FalkorDB and return the graph results.
        """
        logger.info("🔹 Executing graph queries...")
        
        try:
            # Clear existing graph
            self.graph_db.query("MATCH (n) DETACH DELETE n")
            logger.info("✅ Cleared existing graph")
            
            # Execute creation queries one at a time
            for i, query in enumerate(queries, 1):
                try:
                    if isinstance(query, tuple):
                        query_text, params = query
                        logger.info(f"🔹 Executing query {i}: {query_text}")
                        logger.info(f"🔹 Query params: {params}")
                        result = self.graph_db.query(query_text, params)
                        logger.info(f"🔹 Query result: {result}")
                    else:
                        logger.info(f"🔹 Executing query {i}: {query}")
                        result = self.graph_db.query(query)
                        logger.info(f"🔹 Query result: {result}")
                    logger.info(f"✅ Completed query {i}/{len(queries)}")
                except Exception as e:
                    logger.error(f"❌ Failed to execute query {i}: {str(e)}")
                    logger.error(f"Query: {query}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Query all nodes and relationships
            logger.info("🔹 Querying all nodes...")
            nodes_result = self.graph_db.query("MATCH (n) RETURN n, labels(n) as labels, properties(n) as props")
            logger.info(f"🔹 Found {len(nodes_result.result_set)} nodes")
            
            logger.info("🔹 Querying all edges...")
            edges_result = self.graph_db.query("""
                MATCH (source)-[r]->(target)
                RETURN type(r) as type,
                       properties(r) as props,
                       source.name as source,
                       target.name as target,
                       labels(source) as sourceLabels,
                       labels(target) as targetLabels,
                       id(r) as id
            """)
            logger.info(f"🔹 Found {len(edges_result.result_set)} edges")
            
            # Log the raw results
            logger.info(f"🔹 Raw nodes result: {nodes_result.result_set}")
            logger.info(f"🔹 Raw edges result: {edges_result.result_set}")
            
            # Convert results to dictionary
            nodes = []
            edges = []
            
            # Create a map of node names to their indices
            node_indices = {}
            
            for i, record in enumerate(nodes_result.result_set):
                node = {
                    "id": i,  # Add unique ID for each node
                    "labels": record[1],
                    "properties": record[2]
                }
                nodes.append(node)
                node_indices[record[2].get('name')] = i
                logger.info(f"🔹 Processed node {i}: {json.dumps(node, indent=2)}")
            
            for record in edges_result.result_set:
                source_name = record[2]
                target_name = record[3]
                
                # Skip if either source or target is not found
                if source_name not in node_indices or target_name not in node_indices:
                    logger.warning(f"⚠️ Skipping edge: Invalid source or target node (source={source_name}, target={target_name})")
                    continue
                
                edge = {
                    "id": record[6],  # Add edge ID
                    "type": record[0],
                    "properties": record[1],
                    "source": node_indices[source_name],  # Use node indices
                    "target": node_indices[target_name],
                    "sourceNode": source_name,
                    "targetNode": target_name
                }
                edges.append(edge)
                logger.info(f"🔹 Processed edge: {json.dumps(edge, indent=2)}")
            
            logger.info(f"✅ Retrieved graph with {len(nodes)} nodes and {len(edges)} edges")
            return {
                "nodes": nodes,
                "edges": edges
            }
            
        except Exception as e:
            logger.error(f"❌ Error executing graph queries: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"nodes": [], "edges": []}

    def _format_graph_for_ui(self, nodes, edges):
        """
        Format the graph data for UI display.
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            
        Returns:
            A dictionary with nodes and edges in the format expected by the UI
        """
        ui_nodes = []
        ui_edges = []
        
        # Process nodes to match UI expectations
        for node in nodes:
            ui_node = {
                "id": node["id"],
                "title": node["properties"].get("name", "Unknown"),
                "type": node["properties"].get("category", "Entity"),
                "summary": node["properties"].get("description", ""),
                "labels": node["labels"]
            }
            ui_nodes.append(ui_node)
        
        # Process edges to match UI expectations
        for edge in edges:
            ui_edge = {
                "source": edge["source"],
                "target": edge["target"],
                "relation": edge["type"],
                "properties": edge["properties"]
            }
            ui_edges.append(ui_edge)
        
        return {
            "nodes": ui_nodes,
            "edges": ui_edges
        }

    def generate_ontology_graph(self, markdown_content: str) -> Dict[str, Any]:
        """
        Generate an ontology graph from markdown content.
        
        Args:
            markdown_content: The markdown content to process
            
        Returns:
            A dictionary containing the ontology graph structure
        """
        logger.info("\n=== ONTOLOGY GRAPH GENERATION ===")
        logger.info(f"📝 Received markdown content length: {len(markdown_content)} characters")
        
        try:
            # Extract entities and relations
            logger.info("🔹 Extracting entities and relations...")
            data = self.extract_entities_and_relations(markdown_content)
            
            # Create graph queries
            queries = self.create_graph_queries(data["entities"], data["relations"])
            
            # Execute queries and get graph result
            graph_result = self.execute_graph_queries(queries)
            
            # Format the result for UI
            ui_formatted = self._format_graph_for_ui(graph_result["nodes"], graph_result["edges"])
            
            logger.info(f"📝 Generated ontology graph with {len(data['entities'])} entities and {len(data['relations'])} relations")
            logger.info("=== END ONTOLOGY GRAPH GENERATION ===\n")
            
            return ui_formatted
            
        except Exception as e:
            logger.error(f"❌ Error generating ontology graph: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"nodes": [], "edges": []} 