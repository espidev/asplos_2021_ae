ptxas -arch=sm_70 -m64  "pagerank.ptx"  -o "pagerank.sm_70.cubin" 
fatbinary --create="pagerank.fatbin" -64 "--image=profile=sm_70,file=pagerank.sm_70.cubin" "--image=profile=compute_70,file=pagerank.ptx" --embedded-fatbin="pagerank.fatbin.c" 
gcc -E -x c++ -D__CUDACC__ -D__NVCC__  -O3 -I"/home/tgrogers-raid/a/aalawneh/../common/NVIDIA_GPU_Computing_SDK/10.1/../4.2/C/common/inc" -I"/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/include" "-I/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/..//include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=168 -include "cuda_runtime.h" -m64 "pagerank.cu" > "pagerank.cpp4.ii" 
cudafe++ --c++14 --gnu_version=70500 --allow_managed  --m64 --parse_templates --gen_c_file_name "pagerank.cudafe1.cpp" --stub_file_name "pagerank.cudafe1.stub.c" --module_id_file_name "pagerank.module_id" "pagerank.cpp4.ii" 
gcc -D__CUDA_ARCH__=700 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -O3 -I"/home/tgrogers-raid/a/aalawneh/../common/NVIDIA_GPU_Computing_SDK/10.1/../4.2/C/common/inc" -I"/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/include" "-I/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/..//include"   -m64 -o "pagerank.o" "pagerank.cudafe1.cpp" 
nvlink --arch=sm_70 --register-link-binaries="PR_TP_dlink.reg.c"  -m64 -L"/home/tgrogers-raid/a/aalawneh/../common/NVIDIA_GPU_Computing_SDK/10.1/../4.2/C/lib" -lcutil_x86_64 -lcudart   "-L/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/..//lib64/stubs" "-L/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/..//lib64" -cpu-arch=X86_64 "pagerank.o" "parse.o" "util.o"  -lcudadevrt  -o "PR_TP_dlink.sm_70.cubin"
fatbinary --create="PR_TP_dlink.fatbin" -64 -link "--image=profile=sm_70,file=PR_TP_dlink.sm_70.cubin" --embedded-fatbin="PR_TP_dlink.fatbin.c" 
gcc -c -x c++ -DFATBINFILE="\"PR_TP_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"PR_TP_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  -O3 -I"/home/tgrogers-raid/a/aalawneh/../common/NVIDIA_GPU_Computing_SDK/10.1/../4.2/C/common/inc" -I"/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/include" "-I/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/..//include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=168 -m64 -o "PR_TP_dlink.o" "/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/crt/link.stub" 
g++ -O3 -m64 -o "../../../../bin/10.1/release/PR_TP" -Wl,--start-group "PR_TP_dlink.o" "pagerank.o" "parse.o" "util.o" -L"/home/tgrogers-raid/a/aalawneh/../common/NVIDIA_GPU_Computing_SDK/10.1/../4.2/C/lib" -lcutil_x86_64 -lcudart   "-L/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/..//lib64/stubs" "-L/home/tgrogers-raid/a/aalawneh/../common/cuda-10.1/bin/..//lib64" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group 