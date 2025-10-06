#!/bin/bash
# Comparison visualization - Compare before/after optimization

# First, generate baseline and optimized results
# ./tools/shocdriver.py --backend cuda --size 2 --output baseline.csv
# ./tools/shocdriver.py --backend cuda --size 2 --output optimized.csv

# Then create visualizations
./tools/shocviz.py baseline.csv optimized.csv \
  --compare \
  --output reports/comparison.png

./tools/shocviz.py baseline.csv optimized.csv \
  --speedup \
  --output reports/speedup.png

echo "Comparison charts saved to reports/"
