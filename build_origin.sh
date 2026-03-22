#!/usr/bin/env bash
# Build script for original speechballoon C++ code from src_origin
# Usage: arch -x86_64 ./build_origin.sh

set -e
SRC_DIR="src_origin"
OUT_BIN="speechballoon_detector_origin"

# Find all cpp files in src_origin
CPP_FILES=$(find ${SRC_DIR} -name "*.cpp" 2>/dev/null)

if [ -z "$CPP_FILES" ]; then
    echo "No .cpp files found in ${SRC_DIR}"
    exit 1
fi

echo "Found source files:"
echo "$CPP_FILES"
echo ""

# Compile command for macOS with OpenCV installed via pkg-config
echo "Compiling with OpenCV4..."
g++ -std=c++11 ${CPP_FILES} -o ${OUT_BIN} `pkg-config --cflags --libs opencv4`

echo "Built ${OUT_BIN}"

# Make executable
chmod +x ${OUT_BIN}

echo "Done."
