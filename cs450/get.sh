#!/bin/bash

cd util/correlation
./get_hw_stats.py -c -D 0 -B oo-get-stat # TODO we can add -p to disable nvprof and use nsight
cd ../../../