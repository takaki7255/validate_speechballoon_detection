#!/usr/bin/env bash
# Build script for speechballoon C++ detector
# Usage: ./build_cpp.sh

set -e
SRC_DIR="src"
OUT_BIN="speechballoon_detector"

# Find all cpp files in src
CPP_FILES=$(ls ${SRC_DIR}/*.cpp)

# Example compile command for macOS with OpenCV installed via pkg-config
echo "Compiling: g++ -std=c++11 ${CPP_FILES} -o ${OUT_BIN} `pkg-config --cflags --libs opencv4`"
g++ -std=c++11 ${CPP_FILES} -o ${OUT_BIN} `pkg-config --cflags --libs opencv4`

echo "Built ${OUT_BIN}"

# Make executable
chmod +x ${OUT_BIN}

echo "Done."