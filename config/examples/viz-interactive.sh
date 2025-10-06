#!/bin/bash
# Interactive HTML visualization with plotly

# Requires: pip install plotly

./tools/shocviz.py results.json \
  --chart bar \
  --interactive \
  --output reports/interactive.html

echo "Interactive report saved to reports/interactive.html"
echo "Open in browser to explore results interactively"
