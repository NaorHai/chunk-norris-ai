import logging
import os

# Configure logging
logger = logging.getLogger(__name__)

def load_prompts_from_mustache(template_file_path, replacements=None):
    """
    Load system and user prompts from a mustache template file.
    
    Args:
        template_file_path: Path to the mustache template file
        replacements: Dictionary of replacements for placeholders in the prompt
                     (e.g., {"text_content": "some text"})
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = ""
    user_prompt = ""
    
    try:
        # Get the absolute path of the current script for relative path resolution
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        
        # Check if file exists
        if not os.path.exists(template_file_path):
            # Try with project root-relative path
            possible_paths = [
                template_file_path,
                os.path.join(project_root, template_file_path),
                os.path.join(project_root, "prompts", os.path.basename(template_file_path)),
                os.path.join(current_dir, "..", "..", "prompts", os.path.basename(template_file_path)),
                os.path.join("prompts", os.path.basename(template_file_path)),
                os.path.join("..", "prompts", os.path.basename(template_file_path)),
                os.path.join("..", "..", "prompts", os.path.basename(template_file_path))
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    template_file_path = path
                    logger.info(f"Found template at: {path}")
                    break
            else:
                error_msg = f"Template file not found. Tried paths: {possible_paths}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        # Load the template file
        with open(template_file_path, 'r', encoding='utf-8') as template_file:
            template_content = template_file.read()
            
            # Extract system prompt (between first {{! System prompt... }} and {{! User prompt... }})
            system_start = template_content.find("{{! System prompt")
            user_start = template_content.find("{{! User prompt")
            
            if system_start == -1:
                error_msg = f"System prompt section not found in template {template_file_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            if user_start == -1:
                error_msg = f"User prompt section not found in template {template_file_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Extract system prompt (skip the comment line)
            system_prompt_with_comment = template_content[system_start:user_start].strip()
            system_prompt_lines = system_prompt_with_comment.split('\n')
            system_prompt = '\n'.join(system_prompt_lines[1:]).strip()
            
            if not system_prompt:
                error_msg = f"System prompt is empty in template {template_file_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Extract user prompt (skip the comment line)
            user_prompt_with_comment = template_content[user_start:].strip()
            user_prompt_lines = user_prompt_with_comment.split('\n')
            user_prompt = '\n'.join(user_prompt_lines[1:]).strip()
            
            if not user_prompt:
                error_msg = f"User prompt is empty in template {template_file_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Apply replacements if provided
            if replacements and isinstance(replacements, dict):
                for key, value in replacements.items():
                    placeholder = f"{{{{{key}}}}}"
                    if placeholder not in user_prompt:
                        logger.warning(f"Placeholder {placeholder} not found in user prompt of {template_file_path}")
                    user_prompt = user_prompt.replace(placeholder, str(value))
        
        return system_prompt, user_prompt
    
    except Exception as e:
        logger.error(f"Error loading prompt template {template_file_path}: {str(e)}")
        return None, None 