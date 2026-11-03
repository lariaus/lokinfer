#!/bin/sh
cd "$(dirname "$0")"
clang++ -std=c++20 -O2 -fobjc-arc -I metal-cpp bench.cpp -framework Metal -framework Foundation -framework QuartzCore -o bench
