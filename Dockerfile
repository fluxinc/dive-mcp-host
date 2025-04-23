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

# Set build arguments for the MCP server repository
ARG MCP_REPO_URL=github.com/fluxinc/rag-mcp-server.git
ARG MCP_REPO_BRANCH=master
ARG GITHUB_TOKEN

# Clone the MCP server repository using GitHub token for authentication
RUN echo "Cloning MCP server repository..."
RUN git clone --branch $MCP_REPO_BRANCH https://${GITHUB_TOKEN:+${GITHUB_TOKEN}@}${MCP_REPO_URL} /app/mcp-server

# Install MCP Node.js dependencies and build
WORKDIR /app/mcp-server

RUN npm install
RUN npm run build
RUN npm prune --production
RUN mkdir -p /app/mcp-server/logs

# Now, go back to main directory and copy the Python application code
WORKDIR /app
COPY . .

# Create README.md file if it doesn't exist
RUN test -f README.md || echo "# Dive MCP Host\n\nPython server component for the Dive application." > README.md

# Install the package with uv in editable mode as specified in README
RUN pip install -e ".[dev]"

# Create a startup script
RUN echo '#!/bin/bash\n\
# Start the Python service\n\
cd /app && dive_httpd --listen 0.0.0.0\n' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose the port
EXPOSE 61990

# Set environment for better Python output
ENV PYTHONUNBUFFERED=1

# Run both services using the startup script
CMD ["/app/start.sh"]