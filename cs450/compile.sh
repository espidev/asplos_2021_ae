#!/bin/bash

cd benchmarks/src/
source setup_environment
make oop_bench -j$(nproc --all) # Change depending on # of cores
# make oop-mem-2 -j$(nproc --all) # Build our new thing directly
cd ../../