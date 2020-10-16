# This bash script formats gpgpusim_simulation using clang-format
THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
clang-format -i ${THIS_DIR}/benchmarks/src/cuda/oop/*/*.cu
clang-format -i ${THIS_DIR}/benchmarks/src/cuda/oop/*/*.h
clang-format -i ${THIS_DIR}/benchmarks/src/cuda/oop-mem/*/*.cu
clang-format -i ${THIS_DIR}/benchmarks/src/cuda/oop-mem/*/*.h
clang-format -i ${THIS_DIR}/benchmarks/src/cuda/mem_alloc/*
