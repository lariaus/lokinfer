#!/bin/sh
# Runs every test suite: C++ unit/E2E (Catch2), compiler passes (lit +
# filecheck), Python (pytest). Usage: scripts/test.sh [build-dir]
set -e
cd "$(dirname "$0")/.."
BUILD=${1:-build}
cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=${BUILD_TYPE:-Debug} >/dev/null
cmake --build "$BUILD"
"$BUILD"/lokinfer_unit_tests
LOKINFER_BUILD_DIR="$PWD/$BUILD" .venv/bin/lit -q tests/lit
.venv/bin/python -m pytest -q
