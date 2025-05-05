"""
Configuration file containing API keys and other sensitive information.
This file is excluded from version control by .gitignore.
"""

# OpenAI API configuration
OPENAI_API_KEY = "your-api-key-here"

# OpenAI Models
GPT4O_MODEL = "gpt-4o"
GPT4O_MINI_MODEL = "gpt-4o-mini"
DEFAULT_MODEL = GPT4O_MINI_MODEL

# Model parameters
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 4000

# Directory paths
UPLOADS_DIR = "uploads"
IMAGES_DIR = "uploads/images"
RTF_PREVIEWS_DIR = "static/rtf_previews"

# Any other API keys or configuration settings
# EXAMPLE_API_KEY = "your-example-api-key-here" 