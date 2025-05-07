FROM python:3.12-slim

WORKDIR /app

# Install system dependencies, Git, and Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    postgresql-client \
    curl \
    gnupg \
    git \
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

# Install the package with pip in editable mode with caching
# Use BuildKit cache mount to persist pip cache between builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[dev]" && \
    pip install watchdog[watchmedo]

# Create a startup script that ensures the SQLite database exists
RUN echo '#!/bin/bash\n\
# Ensure the database directory exists with correct permissions\n\
mkdir -p /app\n\
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
# Start the Python service with hot reloading\n\
cd /app && watchmedo auto-restart --directory=/app/dive_mcp_host --pattern="*.py" --recursive -- dive_httpd --listen 0.0.0.0\n\
' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose the port
EXPOSE 61990

# Set environment for better Python output
ENV PYTHONUNBUFFERED=1

# Run both services using the startup script
CMD ["/app/start.sh"]