ptxas -arch=sm_70 -m64  "nbody.ptx"  -o "nbody.sm_70.cubin" 
fatbinary --create="nbody.fatbin" -64 "--image=profile=sm_70,file=nbody.sm_70.cubin" "--image=profile=compute_70,file=nbody.ptx" --embedded-fatbin="nbody.fatbin.c" 
gcc -E -x c++ -D__CUDACC__ -D__NVCC__  -O3 -I"/home/tgrogers-raid/a/aalawneh/../common/NVIDIA_GPU_Computing_SDK/10.1/../4.2/C/common/inc" -I"/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/include" "-I/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/..//include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=168 -include "cuda_runtime.h" -m64 "nbody.cu" > "nbody.cpp4.ii" 
cudafe++ --c++14 --gnu_version=70500 --allow_managed  --m64 --parse_templates --gen_c_file_name "nbody.cudafe1.cpp" --stub_file_name "nbody.cudafe1.stub.c" --module_id_file_name "nbody.module_id" "nbody.cpp4.ii" 
gcc -D__CUDA_ARCH__=700 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -O3 -I"/home/tgrogers-raid/a/aalawneh/../common/NVIDIA_GPU_Computing_SDK/10.1/../4.2/C/common/inc" -I"/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/include" "-I/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/..//include"   -m64 -o "nbody.o" "nbody.cudafe1.cpp" 
nvlink --arch=sm_70 --register-link-binaries="nbodyV_TP_dlink.reg.c"  -m64 -L"/home/tgrogers-raid/a/aalawneh/../common/NVIDIA_GPU_Computing_SDK/10.1/../4.2/C/lib" -lcutil_x86_64 -lcudart   "-L/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/..//lib64/stubs" "-L/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/..//lib64" -cpu-arch=X86_64 "nbody.o"  -lcudadevrt  -o "nbodyV_TP_dlink.sm_70.cubin"
fatbinary --create="nbodyV_TP_dlink.fatbin" -64 -link "--image=profile=sm_70,file=nbodyV_TP_dlink.sm_70.cubin" --embedded-fatbin="nbodyV_TP_dlink.fatbin.c" 
gcc -c -x c++ -DFATBINFILE="\"nbodyV_TP_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"nbodyV_TP_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  -O3 -I"/home/tgrogers-raid/a/aalawneh/../common/NVIDIA_GPU_Computing_SDK/10.1/../4.2/C/common/inc" -I"/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/include" "-I/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/..//include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=168 -m64 -o "nbodyV_TP_dlink.o" "/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/crt/link.stub" 
g++ -O3 -m64 -o "/home/tgrogers-raid/a/aalawneh/gpgpu-sim_simulations_chucnk/benchmarks/src/../bin/10.1/release/nbodyV_TP" -Wl,--start-group "nbodyV_TP_dlink.o" "nbody.o" -L"/home/tgrogers-raid/a/aalawneh/../common/NVIDIA_GPU_Computing_SDK/10.1/../4.2/C/lib" -lcutil_x86_64 -lcudart   "-L/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/..//lib64/stubs" "-L/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/..//lib64" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group 
