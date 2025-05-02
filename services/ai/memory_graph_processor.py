import os
import json
import requests
from typing import Dict, Any, Optional

from config import OPENAI_API_KEY, GPT4O_MINI_MODEL, DEFAULT_TEMPERATURE


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
    
    def generate_memory_graph(self, markdown_content: str) -> Dict[str, Any]:
        """
        Generate a memory graph from markdown content
        
        Args:
            markdown_content: The markdown content to analyze
            
        Returns:
            A dictionary containing the memory graph structure with nodes and edges
        """
        # Load the memory graph prompt template
        prompt_template = self._load_prompt_template()
        
        # Replace placeholder with actual markdown content
        prompt = prompt_template.replace("{{markdown_content}}", markdown_content)
        
        # Call OpenAI API to generate the memory graph
        memory_graph_json = self._call_openai_api(prompt)
        
        # Parse and return the memory graph
        return self._parse_memory_graph(memory_graph_json)
    
    def _load_prompt_template(self) -> str:
        """
        Load the memory graph prompt template from file
        
        Returns:
            The prompt template as a string
        """
        template_path = os.path.join("prompts", "memory_graph_prompt.mustache")
        try:
            with open(template_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            print(f"Error loading memory graph prompt template: {str(e)}")
            # Fallback to a minimal prompt if the file can't be loaded
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
            print("\n=== SENDING LLM REQUEST (Memory Graph Generation) ===")
            print(f"Model: {self.model}")
            print(f"Input text length: {len(prompt)} characters")
            print(f"Temperature: {self.temperature}")
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code != 200:
                error_msg = f"API Error: {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f" - {error_data['error'].get('message', 'Unknown error')}"
                    print(f"Full error response: {error_data}")
                except Exception:
                    error_msg += f" - {response.text[:200]}..."
                
                print(error_msg)
                return '{}'
                
            result = response.json()
            
            print(f"Response received, status code: {response.status_code}")
            print(f"Tokens used: {result.get('usage', {}).get('total_tokens', 'N/A')}")
            print("=== END OF LLM REQUEST ===\n")
            
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            # For debugging only
            import traceback
            print(f"Exception traceback: {traceback.format_exc()}")
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
            # Extract the JSON part from the response
            # The response might contain additional text before or after the JSON
            start_idx = memory_graph_json.find('{')
            end_idx = memory_graph_json.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = memory_graph_json[start_idx:end_idx]
                memory_graph = json.loads(json_str)
                return memory_graph
            else:
                print("Invalid memory graph JSON format")
                return {"nodes": [], "edges": []}
                
        except json.JSONDecodeError as e:
            print(f"Error parsing memory graph JSON: {str(e)}")
            print(f"Raw JSON: {memory_graph_json}")
            return {"nodes": [], "edges": []}
        except Exception as e:
            print(f"Unexpected error parsing memory graph: {str(e)}")
            return {"nodes": [], "edges": []} 