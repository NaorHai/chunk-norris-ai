#!/bin/bash

# Define colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}   Chuck Norris AI - Startup Script     ${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Default configuration
PORT=8080
DEBUG=false
HOST="0.0.0.0"

# Function to show usage
show_help() {
    echo "Usage: ./start.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  -p, --port PORT     Specify the port to run the app on (default: 8080)"
    echo "  -d, --debug         Enable debug mode"
    echo "  -h, --host HOST     Specify the host to bind to (default: 0.0.0.0)"
    echo "  --help              Show this help message and exit"
    echo
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                PORT="$2"
                shift 2
            else
                echo -e "${RED}Error: Port number is required for -p|--port${NC}"
                exit 1
            fi
            ;;
        -d|--debug)
            DEBUG=true
            shift
            ;;
        -h|--host)
            if [[ -n "$2" ]]; then
                HOST="$2"
                shift 2
            else
                echo -e "${RED}Error: Host is required for -h|--host${NC}"
                exit 1
            fi
            ;;
        --help)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            ;;
    esac
done

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment. Please install python3-venv package.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Virtual environment created successfully.${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment.${NC}"
    exit 1
fi
echo -e "${GREEN}Virtual environment activated.${NC}"

# Check if requirements are installed
echo -e "${BLUE}Checking for required packages...${NC}"
if ! pip list | grep -q "flask"; then
    echo -e "${YELLOW}Required packages not found. Installing...${NC}"
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install required packages.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Required packages installed successfully.${NC}"
else
    echo -e "${GREEN}Required packages are already installed.${NC}"
fi

# Create uploads directory if not exists
if [ ! -d "uploads" ]; then
    echo -e "${BLUE}Creating uploads directory...${NC}"
    mkdir -p uploads/images
    echo -e "${GREEN}Uploads directory created.${NC}"
fi

# Create static directory if not exists
if [ ! -d "static/rtf_previews" ]; then
    echo -e "${BLUE}Creating static directories...${NC}"
    mkdir -p static/rtf_previews
    echo -e "${GREEN}Static directories created.${NC}"
fi

# Run falkor_schema_loader.py first
echo -e "\n${BLUE}Running FalkorDB schema loader...${NC}"
python3 falkor_schema_loader.py
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}FalkorDB schema loader failed, but continuing with app startup...${NC}"
else
    echo -e "${GREEN}FalkorDB schema loaded successfully.${NC}"
fi

# Set the correct port in app.py
echo -e "${BLUE}Starting Flask application on port ${PORT}...${NC}"
# Check if config.py exists, create it if not
if [ ! -f "config.py" ]; then
    echo -e "${YELLOW}config.py not found. Creating default configuration...${NC}"
    echo "OPENAI_API_KEY = ''" > config.py
    echo -e "${YELLOW}Please update config.py with your OpenAI API key.${NC}"
fi

# Construct arguments for the Flask application
APP_ARGS="--port $PORT --host $HOST"
if [ "$DEBUG" = true ]; then
    APP_ARGS="$APP_ARGS --debug"
    echo -e "${YELLOW}Running in debug mode${NC}"
fi

# Start the Flask application
echo -e "${BLUE}Running: python3 app.py $APP_ARGS${NC}"
python3 app.py $APP_ARGS
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to start Flask application.${NC}"
    exit 1
fi 