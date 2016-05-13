#!/bin/bash
echo "Running level0 hip benchmarks"
./src/hip/level0/epmpi/BusSpeedDownload
./src/hip/level0/epmpi/BusSpeedReadback
./src/hip/level0/epmpi/DeviceMemory
./src/hip/level0/epmpi/MaxFlops
