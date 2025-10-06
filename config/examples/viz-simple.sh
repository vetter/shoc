#!/bin/bash
# Simple visualization example - Generate bar chart from single run

./tools/shocviz.py results.csv \
  --chart bar \
  --output reports/overview.png

echo "Bar chart saved to reports/overview.png"
