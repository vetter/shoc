#!/bin/bash
echo "Building SHOC using AutoMake"
autoreconf -i
./configure
make
