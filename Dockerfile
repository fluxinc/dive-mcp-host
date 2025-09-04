FROM mcr.microsoft.com/playwright:v1.52.0-jammy

WORKDIR /app
ARG DATABRIDGE_SERVER_URL
ARG OPENAI_API_KEY
ARG DIVE_CONFIG_DIR

# Install system dependencies, Git, Node.js, and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    postgresql-client \
    curl \
    gnupg \
    git \
    python3-pip \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install PM2 globally for process management
RUN npm install -g pm2

# Now, go back to main directory and copy the Python application code
COPY . .

# Install MCP Node.js dependencies and build
WORKDIR /app/RAG-mcp-server

RUN npm install
RUN npm run build
RUN npm prune --production
RUN mkdir -p /app/RAG-mcp-server/logs

WORKDIR /app


# Create README.md file if it doesn't exist
RUN test -f README.md || echo "# Dive MCP Host\n\nPython server component for the Dive application." > README.md

# Install uv for Python package management
RUN pip install uv

# Create a startup script that ensures the SQLite database exists
RUN echo '#!/bin/bash\n\
# Ensure the database directory exists with correct permissions\n\
mkdir -p /app\n\
\n\
# Create config directory if it does not exist\n\
mkdir -p $DIVE_CONFIG_DIR\n\
\n\
# Create mcp_config.json if it does not exist\n\
if [ ! -f "$DIVE_CONFIG_DIR/mcp_config.json" ]; then\n\
    cat > "$DIVE_CONFIG_DIR/mcp_config.json" << EOF\n\
{\n\
    "mcpServers": {\n\
        "rag-mcp-server": {\n\
            "transport": "command",\n\
            "command": "node",\n\
            "enabled": true,\n\
            "args": [\n\
                "/app/RAG-mcp-server/dist/index.js",\n\
                "--log"\n\
            ],\n\
            "env": {\n\
                "LOG_FILE_PATH": "/app/RAG-mcp-server/logs/mcp-server.log",\n\
                "DATABRIDGE_URL": "${DATABRIDGE_SERVER_URL}"\n\
            }\n\
        }\n\
    }\n\
}\n\
EOF\n\
fi\n\
\n\
# Create model_config.json if it does not exist\n\
if [ ! -f "$DIVE_CONFIG_DIR/model_config.json" ]; then\n\
    cat > "$DIVE_CONFIG_DIR/model_config.json" << EOF\n\
{\n\
  "activeProvider": "openai",\n\
  "configs": {\n\
    "openai": {\n\
      "modelProvider": "openai",\n\
      "model": "gpt-4o-mini",\n\
      "apiKey": "${OPENAI_API_KEY}",\n\
      "base_url": "https://api.openai.com/v1",\n\
      "temperature": 0.2,\n\
      "top_p": 0.5\n\
    }\n\
  },\n\
  "enable_tools": true\n\
}\n\
EOF\n\
fi\n\
\n\
# Ensure SQLite database file exists and is not a directory\n\
if [ -d "/app/db.sqlite" ]; then\n\
    echo "Error: /app/db.sqlite is a directory, not a file. Removing it."\n\
    rm -rf /app/db.sqlite\n\
fi\n\
\n\
if [ ! -f "/app/db.sqlite" ]; then\n\
    echo "Creating empty SQLite database file"\n\
    touch /app/db.sqlite\n\
    chmod 666 /app/db.sqlite\n\
fi\n\
\n\
# Start the Python service\n\
cd /app && uv run dive_httpd --listen 0.0.0.0 --port 61990\n\
' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose the port
EXPOSE 61990

# Set environment for better Python output
ENV PYTHONUNBUFFERED=1

# Run both services using the startup script
CMD ["/app/start.sh"]