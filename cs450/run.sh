#/bin/bash

cd util/correlation
# time ./run_hw.py -c -R 1 -D 0 -B oo-cuda,oo-concord,oo-mem,oo-coal,oo-tp,oo-mem-2
time ./run_hw.py -c -R 1 -D 0 -B oo-mem-2
cd ../../