import os
import json
import requests
import time
from typing import Dict, Any, Optional
import logging

from config import OPENAI_API_KEY, GPT4O_MINI_MODEL, DEFAULT_TEMPERATURE

# Get logger for this module
logger = logging.getLogger(__name__)

class MemoryGraphProcessor:
    """
    Processor that generates a memory graph from markdown content using OpenAI's API
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the memory graph processor
        
        Args:
            api_key: OpenAI API key (defaults to config value)
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = GPT4O_MINI_MODEL
        self.temperature = DEFAULT_TEMPERATURE
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        logger.info(f"🔹 Initializing MemoryGraphProcessor with model={self.model}")
    
    def generate_memory_graph(self, markdown_content: str) -> Dict[str, Any]:
        """
        Generate a memory graph from markdown content
        
        Args:
            markdown_content: The markdown content to analyze
            
        Returns:
            A dictionary containing the memory graph structure with nodes and edges
        """
        logger.info("\n=== MEMORY GRAPH GENERATION ===")
        logger.info(f"📝 Received markdown content length: {len(markdown_content)} characters")
        start_time = time.time()
        
        try:
            # Load the memory graph prompt template
            logger.info("🔹 Loading memory graph prompt template...")
            load_start = time.time()
            prompt_template = self._load_prompt_template()
            load_time = time.time() - load_start
            logger.info(f"✅ Prompt template loaded in {load_time:.2f} seconds")
            
            # Replace placeholder with actual markdown content
            logger.info("🔹 Preparing prompt with markdown content...")
            prompt = prompt_template.replace("{{markdown_content}}", markdown_content)
            
            # Call OpenAI API to generate the memory graph
            logger.info("🔹 Calling OpenAI API to generate memory graph...")
            api_start = time.time()
            memory_graph_json = self._call_openai_api(prompt)
            api_time = time.time() - api_start
            logger.info(f"✅ OpenAI API call completed in {api_time:.2f} seconds")
            
            # Parse and return the memory graph
            logger.info("🔹 Parsing memory graph response...")
            parse_start = time.time()
            memory_graph = self._parse_memory_graph(memory_graph_json)
            parse_time = time.time() - parse_start
            logger.info(f"✅ Memory graph parsed in {parse_time:.2f} seconds")
            
            # Log the results
            node_count = len(memory_graph.get('nodes', []))
            edge_count = len(memory_graph.get('edges', []))
            logger.info(f"📝 Memory graph contains {node_count} nodes and {edge_count} edges")
            
            return memory_graph
        except Exception as e:
            logger.error(f"❌ Error in memory graph generation: {str(e)}")
            import traceback
            logger.error(f"❌ Exception traceback: {traceback.format_exc()}")
            return {"nodes": [], "edges": []}
        finally:
            elapsed_time = time.time() - start_time
            logger.info(f"⏱️ Memory graph generation completed in {elapsed_time:.2f} seconds")
            logger.info("=== END MEMORY GRAPH GENERATION ===\n")
    
    def _load_prompt_template(self) -> str:
        """
        Load the memory graph prompt template from file
        
        Returns:
            The prompt template as a string
        """
        template_path = os.path.join("prompts", "memory_graph_prompt.mustache")
        try:
            with open(template_path, "r", encoding="utf-8") as file:
                template = file.read()
                logger.info(f"✅ Memory graph prompt template loaded from {template_path}")
                return template
        except Exception as e:
            logger.error(f"❌ Error loading memory graph prompt template: {str(e)}")
            # Fallback to a minimal prompt if the file can't be loaded
            logger.warning("⚠️ Using fallback prompt template")
            return """
            You are a document intelligence system that builds structured memory graphs from markdown content.
            Given the content, extract and organize key information units into a hierarchical graph.
            
            Input Markdown:
            {{markdown_content}}
            
            Output Memory Graph as a JSON object with nodes and edges arrays.
            """
    
    def _call_openai_api(self, prompt: str) -> str:
        """
        Call OpenAI API to generate the memory graph
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The generated memory graph as a JSON string
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a document intelligence system that extracts structured memory graphs from content."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature
        }
        
        try:
            logger.info("\n=== SENDING LLM REQUEST (Memory Graph Generation) ===")
            logger.info(f"🔹 Model: {self.model}")
            logger.info(f"🔹 Input text length: {len(prompt)} characters")
            logger.info(f"🔹 Temperature: {self.temperature}")
            logger.info(f"🔹 Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            api_start = time.time()
            response = requests.post(self.api_url, headers=headers, json=payload)
            api_time = time.time() - api_start
            
            if response.status_code != 200:
                error_msg = f"API Error: {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f" - {error_data['error'].get('message', 'Unknown error')}"
                    logger.error(f"Full error response: {error_data}")
                except Exception:
                    error_msg += f" - {response.text[:200]}..."
                
                logger.error(error_msg)
                return '{}'
                
            result = response.json()
            
            logger.info(f"✅ Response received, status code: {response.status_code}")
            logger.info(f"✅ API request completed in {api_time:.2f} seconds")
            logger.info(f"📊 Tokens used: {result.get('usage', {}).get('total_tokens', 'N/A')}")
            logger.info(f"  - Prompt tokens: {result.get('usage', {}).get('prompt_tokens', 'N/A')}")
            logger.info(f"  - Completion tokens: {result.get('usage', {}).get('completion_tokens', 'N/A')}")
            logger.info("=== END OF LLM REQUEST ===\n")
            
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"❌ Error calling OpenAI API: {str(e)}")
            # For debugging only
            import traceback
            logger.error(f"❌ Exception traceback: {traceback.format_exc()}")
            return '{}'
    
    def _parse_memory_graph(self, memory_graph_json: str) -> Dict[str, Any]:
        """
        Parse the memory graph JSON string into a dictionary
        
        Args:
            memory_graph_json: The memory graph JSON string
            
        Returns:
            A dictionary containing the parsed memory graph
        """
        try:
            # Log the raw response for debugging
            logger.info(f"📝 Raw LLM response:\n{memory_graph_json[:500]}...")  # First 500 chars
            
            # Extract the JSON part from the response
            start_idx = memory_graph_json.find('{')
            end_idx = memory_graph_json.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                logger.info(f"✅ Found JSON boundaries: start={start_idx}, end={end_idx}")
                json_str = memory_graph_json[start_idx:end_idx]
                logger.info(f"📝 Extracted JSON string:\n{json_str[:200]}...")  # First 200 chars
                
                try:
                    memory_graph = json.loads(json_str)
                    logger.info(f"✅ Successfully parsed JSON into memory graph object")
                    
                    # Validate and log the structure
                    nodes = memory_graph.get('nodes', [])
                    edges = memory_graph.get('edges', [])
                    
                    logger.info(f"📊 Initial node count: {len(nodes)}")
                    logger.info(f"📊 Initial edge count: {len(edges)}")
                    
                    if not edges:
                        logger.warning("⚠️ No edges found in the initial parsed graph")
                        # Log the first few nodes to check their structure
                        if nodes:
                            logger.info("📝 Sample node structure:")
                            for i, node in enumerate(nodes[:3]):  # Show first 3 nodes
                                logger.info(f"  Node {i}: {json.dumps(node, indent=2)}")
                    
                    # Create a map of node IDs to their indices
                    node_indices = {}
                    for i, node in enumerate(nodes):
                        node_id = node.get('id', i)
                        node_indices[node_id] = i
                        if 'id' not in node:
                            node['id'] = i
                    
                    logger.info(f"📊 Created node index map with {len(node_indices)} entries")
                    
                    # Ensure each edge has source and target properties
                    valid_edges = []
                    for edge in edges:
                        if not isinstance(edge, dict):
                            logger.warning(f"⚠️ Skipping invalid edge (not a dict): {edge}")
                            continue
                            
                        source_id = edge.get('source')
                        target_id = edge.get('target')
                        
                        if source_id is None or target_id is None:
                            # Try to use sourceNode/targetNode if available
                            source_id = edge.get('sourceNode')
                            target_id = edge.get('targetNode')
                            logger.info(f"🔍 Using alternative edge properties: sourceNode={source_id}, targetNode={target_id}")
                        
                        if source_id is not None and target_id is not None:
                            # Convert string IDs to integers if needed
                            try:
                                source_idx = node_indices.get(str(source_id)) or node_indices.get(int(source_id))
                                target_idx = node_indices.get(str(target_id)) or node_indices.get(int(target_id))
                                
                                if source_idx is not None and target_idx is not None:
                                    edge['source'] = source_idx
                                    edge['target'] = target_idx
                                    valid_edges.append(edge)
                                    logger.info(f"✅ Added valid edge: {source_idx} -> {target_idx}")
                                else:
                                    logger.warning(f"⚠️ Skipping edge: Invalid node indices (source={source_idx}, target={target_idx})")
                            except (ValueError, TypeError) as e:
                                logger.warning(f"⚠️ Error converting edge IDs: {str(e)}")
                                continue
                        else:
                            logger.warning(f"⚠️ Skipping edge: Missing source or target: {edge}")
                    
                    memory_graph['edges'] = valid_edges
                    logger.info(f"✅ Final graph: {len(nodes)} nodes and {len(valid_edges)} edges")
                    
                    # Log a sample of the final edges
                    if valid_edges:
                        logger.info("📝 Sample of final edges:")
                        for edge in valid_edges[:3]:  # Show first 3 edges
                            logger.info(f"  Edge: {edge.get('source')} -> {edge.get('target')} ({edge.get('type', 'unknown')})")
                    
                    return memory_graph
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Error parsing JSON: {str(e)}")
                    # Print the problematic section
                    error_pos = str(e).find("char")
                    if error_pos > 0:
                        error_char = int(str(e)[error_pos+5:].split()[0])
                        context_start = max(0, error_char - 50)
                        context_end = min(len(json_str), error_char + 50)
                        logger.error(f"❌ Error context (char {error_char}):\n...{json_str[context_start:context_end]}...")
                    
                    # Try to clean up the JSON string
                    json_str = json_str.replace(',}', '}').replace(',]', ']')
                    json_str = json_str.replace('}{', '},{').replace('][', '],[')
                    try:
                        memory_graph = json.loads(json_str)
                        logger.info("✅ Successfully parsed JSON after cleanup")
                        return memory_graph
                    except json.JSONDecodeError:
                        logger.error("❌ Failed to parse JSON even after cleanup")
                        return {"nodes": [], "edges": []}
            else:
                logger.error("❌ Invalid memory graph JSON format: Could not find valid JSON object")
                return {"nodes": [], "edges": []}
                
        except Exception as e:
            logger.error(f"❌ Unexpected error parsing memory graph: {str(e)}")
            import traceback
            logger.error(f"❌ Exception traceback: {traceback.format_exc()}")
            return {"nodes": [], "edges": []} 