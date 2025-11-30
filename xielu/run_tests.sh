#!/bin/bash

# XIELU Kernel Test Runner
# This script runs the test suite for the XIELU CUDA kernel implementation

set -e  # Exit on error

echo "========================================"
echo "XIELU Kernel Test Suite"
echo "========================================"
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. CUDA may not be available."
fi

# Check if pytest is installed
if ! python -c "import pytest" 2>/dev/null; then
    echo "pytest not found. Installing test dependencies..."
    pip install -r requirements-dev.txt
fi

# Check if xielu module is installed
if ! python -c "import xielu" 2>/dev/null; then
    echo "xielu module not found. Installing..."
    cd xielu && pip install -e . && cd ..
fi

echo ""
echo "Running tests..."
echo "========================================"
echo ""

# Run pytest with various options
# - Use -v for verbose output
# - Use --tb=short for shorter tracebacks
# - Use -x to stop on first failure (optional, remove if you want to run all tests)
# - Use -n auto for parallel execution (if pytest-xdist is installed)

# Basic run
pytest "$@"

# Alternative: Run with coverage (requires pytest-cov)
# pytest --cov=xielu --cov-report=html --cov-report=term "$@"

# Alternative: Run in parallel
# pytest -n auto "$@"

echo ""
echo "========================================"
echo "Tests completed!"
echo "========================================"
