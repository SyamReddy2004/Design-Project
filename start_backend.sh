#!/bin/bash
# Script to launch the AI Grading Backend
echo "Starting NaturaGrade AI Backend..."

# Get directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if venv exists
if [ -d "backend/venv" ]; then
    echo "Activating virtual environment..."
    source backend/venv/bin/activate
else
    echo "Virtual environment not found. Please ensure setup is complete."
    exit 1
fi

# Run the app
echo "Running Flask Server on Port 5000..."
python backend/app/main.py
