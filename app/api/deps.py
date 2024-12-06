from fastapi.security import APIKeyHeader

# Define the header key
API_KEY_NAME = "X-API-Key"

# Create a security dependency for API key
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
