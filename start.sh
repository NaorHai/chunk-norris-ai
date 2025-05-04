#!/bin/bash

# Define colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Success/failure indicators
SUCCESS="✅"
WARNING="⚠️"
ERROR="❌"
INFO="📝"
STEP="🔹"
LOADING="⏳"

# Fixed configuration
APP_PORT=8889
FALKORDB_PORT=6379
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/app.log"

# Print header
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}${BOLD}   Chuck Norris AI - Startup Script     ${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Print startup information
echo -e "${INFO} ${CYAN}Starting Chuck Norris AI at $(date)${NC}"
echo -e "${INFO} ${CYAN}System: $(uname -srm)${NC}"
echo -e "${INFO} ${CYAN}App port: ${APP_PORT}, FalkorDB port: ${FALKORDB_PORT}${NC}"
echo -e "${INFO} ${CYAN}Logs will be saved to: ${LOG_FILE}${NC}\n"

# Create logs directory if it doesn't exist
mkdir -p "${LOG_DIR}"
echo -e "${SUCCESS} Log directory created/verified: ${LOG_DIR}"

# Process script arguments
SHOW_LOGS=false
DEBUG_MODE=false

while getopts "lp:d" option; do
    case $option in
        l) # Enable live log tail
            SHOW_LOGS=true
            ;;
        p) # Set custom port
            APP_PORT=$OPTARG
            ;;
        d) # Enable debug mode
            DEBUG_MODE=true
            ;;
        *) # Invalid option
            echo -e "${ERROR} ${RED}Invalid option${NC}"
            exit 1
            ;;
    esac
done

# Step 1: Setup Python environment
echo -e "${STEP} ${BOLD}Setting up Python environment...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${LOADING} Creating new virtual environment..."
    python3 -m venv venv || { echo -e "${ERROR} ${RED}Failed to create virtual environment${NC}"; exit 1; }
    echo -e "${SUCCESS} Virtual environment created"
else
    echo -e "${INFO} Using existing virtual environment"
fi

echo -e "${LOADING} Activating virtual environment..."
source venv/bin/activate || { echo -e "${ERROR} ${RED}Failed to activate virtual environment${NC}"; exit 1; }
echo -e "${SUCCESS} Virtual environment activated"

# Install packages
echo -e "\n${STEP} ${BOLD}Installing required packages...${NC}"
echo -e "${LOADING} Installing dependencies from requirements.txt..."
pip install -q -r requirements.txt 
if [ $? -eq 0 ]; then
    echo -e "${SUCCESS} All required packages installed successfully"
else
    echo -e "${WARNING} ${YELLOW}Some dependencies may not have installed correctly${NC}"
    echo -e "${INFO} Continuing with available packages"
fi

# Step 2: Setup FalkorDB container
echo -e "\n${STEP} ${BOLD}Setting up FalkorDB container...${NC}"
DOCKER_CONTAINER_NAME="falkordb"
DOCKER_IMAGE="falkordb/falkordb"
FALKORDB_CONNECTED=false

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${WARNING} ${YELLOW}Docker is not installed. Graph functionality will not be available.${NC}"
else
    # Check if port 6379 is already in use
    if lsof -i:${FALKORDB_PORT} > /dev/null 2>&1; then
        echo -e "${ERROR} ${RED}Port ${FALKORDB_PORT} is already in use. Skipping FalkorDB setup.${NC}"
    else
        # Check if container already exists
        if docker ps -a --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER_NAME}$"; then
            echo -e "${INFO} Stopping existing FalkorDB container..."
            docker stop ${DOCKER_CONTAINER_NAME} > /dev/null 2>&1
            docker rm ${DOCKER_CONTAINER_NAME} > /dev/null 2>&1
        fi

        # Create data directory if it doesn't exist
        mkdir -p data
        echo -e "${SUCCESS} Data directory created/verified: data"

        # Run FalkorDB container with the correct parameters
        echo -e "${LOADING} Starting FalkorDB container..."
        docker run -d \
            --name ${DOCKER_CONTAINER_NAME} \
            -p ${FALKORDB_PORT}:6379 \
            -p 3000:3000 \
            -v $(pwd)/data:/data \
            -it --rm \
            ${DOCKER_IMAGE}:latest || { echo -e "${ERROR} ${RED}Failed to start FalkorDB container${NC}"; exit 1; }

        # Wait for container to be ready
        echo -e "${LOADING} Waiting for FalkorDB to be ready..."
        sleep 5

        # Test connection
        if docker exec ${DOCKER_CONTAINER_NAME} redis-cli ping > /dev/null 2>&1; then
            echo -e "${SUCCESS} FalkorDB container is running and accessible"
            FALKORDB_CONNECTED=true
        else
            echo -e "${ERROR} ${RED}Failed to connect to FalkorDB container${NC}"
        fi
    fi
fi

# Print FalkorDB connection status
echo -e "\n${STEP} ${BOLD}FalkorDB Connection Status:${NC}"
if [ "$FALKORDB_CONNECTED" = true ]; then
    echo -e "${SUCCESS} ${GREEN}FalkorDB is connected and ready for graph operations${NC}"
else
    echo -e "${ERROR} ${RED}FalkorDB is not connected - graph functionality will not be available${NC}"
fi

# Step 3: Create required directories
echo -e "\n${STEP} ${BOLD}Creating required directories...${NC}"
mkdir -p uploads/images static/rtf_previews logs
echo -e "${SUCCESS} Required directories created"

# Step 4: Check if app port is in use
echo -e "\n${STEP} ${BOLD}Checking port availability...${NC}"
if lsof -i:${APP_PORT} > /dev/null 2>&1; then
    echo -e "${WARNING} ${YELLOW}Port ${APP_PORT} is already in use.${NC}"
    # Ask if the user wants to kill the process
    echo -e "Process using port ${APP_PORT}:"
    lsof -i:${APP_PORT}
    
    read -p "Kill the process using this port? (y/n): " kill_process
    if [[ "$kill_process" == "y" ]]; then
        # Kill the process
        pid=$(lsof -ti:${APP_PORT})
        kill -9 $pid 2>/dev/null
        echo -e "${SUCCESS} Process killed"
        sleep 1
        echo -e "${LOADING} Checking if port is now available..."
        if lsof -i:${APP_PORT} > /dev/null 2>&1; then
            echo -e "${ERROR} ${RED}Port ${APP_PORT} is still in use after kill attempt${NC}"
            read -p "Enter a new port to use (default: 9090): " new_port
            APP_PORT=${new_port:-9090}
            echo -e "${INFO} Using port ${APP_PORT} instead"
        else
            echo -e "${SUCCESS} Port ${APP_PORT} is now available"
        fi
    else
        # Ask for a new port
        read -p "Enter a new port to use (default: 9090): " new_port
        APP_PORT=${new_port:-9090}
        echo -e "${INFO} Using port ${APP_PORT} instead"
    fi
else
    echo -e "${SUCCESS} Port ${APP_PORT} is available"
fi

# Step 5: Run the Python application
echo -e "\n${STEP} ${BOLD}Starting Flask application...${NC}"
echo -e "${INFO} Starting on port ${APP_PORT}, host 0.0.0.0"

# Configure the app command with proper logging
APP_CMD="python3 app.py --port ${APP_PORT} --host 0.0.0.0"
if [ "$DEBUG_MODE" = true ]; then
    APP_CMD="${APP_CMD} --debug"
    echo -e "${INFO} Running in debug mode with enhanced logging"
fi

# Display a summary of what's happening
echo -e "\n${CYAN}${BOLD}=== STARTUP SUMMARY ====${NC}"
echo -e "${INFO} Python version: $(python3 --version)"
echo -e "${INFO} Virtual environment: $(which python)"
echo -e "${INFO} FalkorDB port: ${FALKORDB_PORT}"
echo -e "${INFO} Application port: ${APP_PORT}"
echo -e "${INFO} Log file: ${LOG_FILE}"
echo -e "${INFO} Starting Flask application: ${APP_CMD}"
echo -e "${CYAN}${BOLD}======================${NC}\n"

# Start the application with proper logging
echo -e "${LOADING} Running: ${APP_CMD}"
if [ "$SHOW_LOGS" = true ]; then
    # Run in foreground with logs displayed
    echo -e "${INFO} Running with live log output"
    ${APP_CMD} 2>&1 | tee -a "${LOG_FILE}"
else
    # Run with proper log redirection
    exec ${APP_CMD} >> "${LOG_FILE}" 2>&1 &
    APP_PID=$!
    echo -e "${SUCCESS} Application started with PID: ${APP_PID}"
    echo -e "${INFO} Logs are being written to: ${LOG_FILE}"
    echo -e "${INFO} To view logs in real-time, run: tail -f ${LOG_FILE}"
    echo -e "${INFO} Application is running at: http://0.0.0.0:${APP_PORT}"
    
    # Wait a moment for app startup
    sleep 2
    
    # Verify app is still running
    if ps -p ${APP_PID} > /dev/null; then
        echo -e "${SUCCESS} Application started successfully"
    else
        echo -e "${ERROR} ${RED}Application failed to start${NC}"
        echo -e "${INFO} Check logs for details: cat ${LOG_FILE}"
        exit 1
    fi
fi

exit 0 