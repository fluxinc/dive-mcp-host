#!/bin/bash

# Run local postgres
docker run -d --rm --name dive-mcp-postgres \
    -e POSTGRES_USER=mcp \
    -e POSTGRES_PASSWORD=mcp \
    -e POSTGRES_DB=mcp \
    -p 5432:5432 \
    postgres:16
