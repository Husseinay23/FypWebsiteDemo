#!/bin/bash
# Script to run the backend server

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the FastAPI server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

