#!/bin/bash
set -e

# Ensure persistent directories exist
mkdir -p /app/downloads /app/vector_database

# Ensure sessions.json exists as a valid JSON file
if [ ! -s /app/sessions.json ]; then
    echo '{}' > /app/sessions.json
fi

exec "$@"