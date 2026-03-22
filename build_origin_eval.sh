#!/usr/bin/env bash
# Build script for evaluation version of original speechballoon C++ code
# Usage: arch -x86_64 ./build_origin_eval.sh

set -e
SRC_DIR="src_origin"
OUT_BIN="speechballoon_detector_origin_eval"

# List of source files (excluding main.cpp, using main_eval.cpp instead)
CPP_FILES="
${SRC_DIR}/main_eval.cpp
${SRC_DIR}/twopage_to_onepage.cpp
${SRC_DIR}/page_classification.cpp
${SRC_DIR}/blackpage_framedetect.cpp
${SRC_DIR}/page_removing_frame.cpp
${SRC_DIR}/frame_separation.cpp
${SRC_DIR}/read_file_path.cpp
${SRC_DIR}/speechballoon_separation.cpp
"

echo "Building evaluation version..."
echo "Source files:"
echo "$CPP_FILES"
echo ""

# Compile command for macOS with OpenCV installed via pkg-config
echo "Compiling with OpenCV4..."
g++ -std=c++11 ${CPP_FILES} -o ${OUT_BIN} `pkg-config --cflags --libs opencv4`

echo "Built ${OUT_BIN}"

# Make executable
chmod +x ${OUT_BIN}

echo "Done."
