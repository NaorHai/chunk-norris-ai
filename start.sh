#!/bin/bash

# Define colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
PORT=8080
DEBUG=false
REDIS_PORT=16379  # Use non-standard port to avoid conflicts

# Print header
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}   Chuck Norris AI - Startup Script     ${NC}"
echo -e "${BLUE}========================================${NC}\n"

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
        --help)
            echo "Usage: ./start.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -p, --port PORT     Specify the port to run the app on (default: 8080)"
            echo "  -d, --debug         Enable debug mode"
            echo "  --help              Show this help message and exit"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Flask port is already in use
check_port() {
    local port=$1
    if command -v nc &> /dev/null; then
        nc -z localhost ${port} 2>/dev/null
        return $?
    elif command -v lsof &> /dev/null; then
        lsof -i:${port} -sTCP:LISTEN &>/dev/null
        return $?
    else
        return 1  # Assume port is free if we can't check
    fi
}

# Handle port conflicts
if check_port ${PORT}; then
    echo -e "${YELLOW}Warning: Port ${PORT} is already in use.${NC}"
    
    # Try to find what's using the port
    if command -v lsof &> /dev/null; then
        echo "Process using port ${PORT}:"
        lsof -i:${PORT} -sTCP:LISTEN
        
        read -p "Kill the process using this port? (y/n): " kill_choice
        if [[ "$kill_choice" =~ ^[Yy]$ ]]; then
            PID=$(lsof -i:${PORT} -sTCP:LISTEN | tail -n 1 | awk '{print $2}')
            echo -e "${YELLOW}Killing process ${PID}...${NC}"
            kill -9 ${PID} || true
            sleep 1
        else
            read -p "Use a different port? (y/n): " port_choice
            if [[ "$port_choice" =~ ^[Yy]$ ]]; then
                read -p "Enter new port: " PORT
            else
                echo -e "${RED}Exiting due to port conflict.${NC}"
                exit 1
            fi
        fi
    else
        read -p "Use a different port? (y/n): " port_choice
        if [[ "$port_choice" =~ ^[Yy]$ ]]; then
            read -p "Enter new port: " PORT
        else
            echo -e "${RED}Exiting due to port conflict.${NC}"
            exit 1
        fi
    fi
fi

# Step 1: Setup Python environment
echo -e "${BLUE}Setting up Python environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv || { echo -e "${RED}Failed to create virtual environment${NC}"; exit 1; }
fi
source venv/bin/activate || { echo -e "${RED}Failed to activate virtual environment${NC}"; exit 1; }

# Install packages
echo -e "${BLUE}Installing essential packages...${NC}"
pip install -q flask flask_cors redis || { echo -e "${RED}Failed to install essential packages${NC}"; exit 1; }

# Step 2: Setup docker container for FalkorDB
echo -e "${BLUE}Setting up FalkorDB container...${NC}"
DOCKER_CONTAINER_NAME="falkordb"
DOCKER_IMAGE="falkordb/falkordb"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker is not installed. Memory graph functionality will not be available.${NC}"
    # Create a dummy falkor_schema_loader.py that always succeeds
    cat > falkor_schema_loader.py << EOF
import sys
print("⚠️ Docker not installed. Memory graph functionality will not be available.")
sys.exit(0)
EOF
else
    # Docker is installed, let's check for Redis conflicts
    # Stop any existing FalkorDB container
    docker stop ${DOCKER_CONTAINER_NAME} 2>/dev/null || true
    docker rm ${DOCKER_CONTAINER_NAME} 2>/dev/null || true
    
    # Check if port is available
    if command -v nc &> /dev/null && nc -z localhost ${REDIS_PORT} 2>/dev/null; then
        echo -e "${YELLOW}Port ${REDIS_PORT} is already in use. Trying another port...${NC}"
        REDIS_PORT=26379  # Try another port
        if command -v nc &> /dev/null && nc -z localhost ${REDIS_PORT} 2>/dev/null; then
            echo -e "${YELLOW}Port ${REDIS_PORT} is also in use. Using random port...${NC}"
            REDIS_PORT=$((10000 + RANDOM % 50000))  # Use random port in high range
        fi
    fi
    
    echo -e "${BLUE}Creating FalkorDB container on port ${REDIS_PORT}...${NC}"
    if ! docker pull ${DOCKER_IMAGE} > /dev/null; then
        echo -e "${YELLOW}Failed to pull FalkorDB image. Using pre-existing image if available.${NC}"
    fi
    
    # Run the container
    if ! docker run -d --name ${DOCKER_CONTAINER_NAME} -p ${REDIS_PORT}:6379 ${DOCKER_IMAGE} > /dev/null; then
        echo -e "${YELLOW}Failed to create FalkorDB container. Memory graph functionality may not work.${NC}"
    else
        echo -e "${GREEN}FalkorDB container is running on port ${REDIS_PORT}${NC}"
        
        # Update the config in falkor_schema_loader.py
        cat > falkor_schema_loader.py << EOF
import time
import sys

print("🚀 Initializing FalkorDB schema...")

# Check if Redis is installed
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    print("❌ Redis module not available. Install with: pip install redis")
    REDIS_AVAILABLE = False
    sys.exit(0)

# Configuration
REDIS_HOST = "localhost"
REDIS_PORT = ${REDIS_PORT}
GRAPH_NAME = "document_graph"

def connect_to_redis():
    """Establish a basic connection to Redis"""
    try:
        # Basic Redis connection first
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        r.ping()  # Test the connection
        print("✅ Connected to FalkorDB/Redis")
        return r
    except Exception as e:
        print(f"❌ Failed to connect to FalkorDB/Redis: {e}")
        return None

def main():
    """Main function to initialize the schema"""
    redis_conn = connect_to_redis()
    if not redis_conn:
        print("⚠️ Cannot continue without Redis connection")
        return False
        
    try:
        # Basic schema setup
        redis_conn.execute_command("GRAPH.QUERY", GRAPH_NAME, "MATCH (n) DETACH DELETE n")
        print("✅ Graph schema initialized")
        return True
    except Exception as e:
        print(f"❌ Schema initialization failed: {e}")
        return False

if __name__ == "__main__":
    time.sleep(2)  # Give Redis a moment to start
    success = main()
    if success:
        print("🎉 FalkorDB schema initialized successfully")
    else:
        print("⚠️ FalkorDB setup failed, but application can still run")
    sys.exit(0)  # Always exit successfully to continue app startup
EOF
    fi
    
    echo -e "${YELLOW}Waiting for FalkorDB to initialize...${NC}"
    sleep 3
fi

# Step 3: Initialize FalkorDB schema
echo -e "${BLUE}Initializing FalkorDB schema...${NC}"
python3 falkor_schema_loader.py
# Always continue regardless of the result

# Step 4: Create required directories
mkdir -p uploads/images static/rtf_previews

# Step 5: Run the Python application
echo -e "${BLUE}Starting Flask application on port ${PORT}...${NC}"

# Assemble command arguments
FLASK_ARGS="--port $PORT --host 0.0.0.0"
if [ "$DEBUG" = true ]; then
    FLASK_ARGS="$FLASK_ARGS --debug"
    echo -e "${YELLOW}Running in debug mode${NC}"
fi

# Start the application
echo -e "${GREEN}Running: python3 app.py ${FLASK_ARGS}${NC}"
# Run with nohup to keep the app running even if the terminal is closed
python3 app.py $FLASK_ARGS

# Exit with the same code as the application
exit $? 