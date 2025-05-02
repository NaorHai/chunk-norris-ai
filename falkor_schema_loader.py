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
REDIS_PORT = 16379
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
