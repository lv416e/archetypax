#!/bin/bash

# Script to run Archetypax tests

# Color definitions
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Running Archetypax tests...${NC}"

# Basic tests
echo -e "${BLUE}Running basic tests...${NC}"
pytest -v

# Generate coverage report
echo -e "${BLUE}Generating coverage report...${NC}"
pytest --cov=archetypax --cov-report=term-missing

# Check results
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed successfully!${NC}"
else
    echo -e "${RED}Tests failed. Please check the errors above.${NC}"
    exit 1
fi

echo -e "${BLUE}Testing complete${NC}"
