#$ _SPACE_= 
#$ _CUDART_=cudart
#$ _HERE_=../../../../../../..//common/cuda-10.1/bin
#$ _THERE_=../../../../../../..//common/cuda-10.1/bin
#$ _TARGET_SIZE_=
#$ _TARGET_DIR_=
#$ _TARGET_SIZE_=64
#$ TOP=../../../../../../..//common/cuda-10.1/bin/..
#$ NVVMIR_LIBRARY_DIR=../../../../../../..//common/cuda-10.1/bin/../nvvm/libdevice
#$ LD_LIBRARY_PATH=../../../../../../..//common/cuda-10.1/bin/../lib:../../../../../../..//common/cuda-10.1/lib64:../../../../../../..//common/cuda-10.1/lib64:
#$ PATH=../../../../../../..//common/cuda-10.1/bin/../nvvm/bin:../../../../../../..//common/cuda-10.1/bin:../../../../../../..//common/cuda-10.1/NsightCompute-2019.3/:../../../../../../..//common/cuda-10.1/bin:/home/tgrogers-raid/a/aalawneh/bin:/home/tgrogers-raid/a/aalawneh/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:.:/opt/thinlinc/bin
#$ INCLUDES="-I../../../../../../..//common/cuda-10.1/bin/..//include"  
#$ LIBRARIES=  "-L../../../../../../..//common/cuda-10.1/bin/..//lib64/stubs" "-L../../../../../../..//common/cuda-10.1/bin/..//lib64"
#$ CUDAFE_FLAGS=
#$ PTXAS_FLAGS=
#$ rm PR_COAL_dlink.reg.c
#$ gcc -D__CUDA_ARCH__=700 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  -O3 "-I../../../../../../..//common/cuda-10.1/bin/..//include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=168 -include "cuda_runtime.h" -m64 "kernel.cu" > "kernel.cpp1.ii" 
#$ cicc --c++14 --gnu_version=70500 --allow_managed   -arch compute_70 -m64 -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "kernel.fatbin.c" -tused -nvvmir-library "../../../../../../..//common/cuda-10.1/bin/../nvvm/libdevice/libdevice.10.bc" --gen_module_id_file --module_id_file_name "kernel.module_id" --orig_src_file_name "kernel.cu" --gen_c_file_name "kernel.cudafe1.c" --stub_file_name "kernel.cudafe1.stub.c" --gen_device_file_name "kernel.cudafe1.gpu"  "kernel.cpp1.ii" -o "kernel.ptx"
#$ ptxas -arch=sm_70 -m64  "kernel.ptx"  -o "kernel.sm_70.cubin" 
#$ fatbinary --create="kernel.fatbin" -64 "--image=profile=sm_70,file=kernel.sm_70.cubin" "--image=profile=compute_70,file=kernel.ptx" --embedded-fatbin="kernel.fatbin.c" 
#$ gcc -E -x c++ -D__CUDACC__ -D__NVCC__  -O3 "-I../../../../../../..//common/cuda-10.1/bin/..//include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=168 -include "cuda_runtime.h" -m64 "kernel.cu" > "kernel.cpp4.ii" 
#$ cudafe++ --c++14 --gnu_version=70500 --allow_managed  --m64 --parse_templates --gen_c_file_name "kernel.cudafe1.cpp" --stub_file_name "kernel.cudafe1.stub.c" --module_id_file_name "kernel.module_id" "kernel.cpp4.ii" 
#$ gcc -D__CUDA_ARCH__=700 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -O3 "-I../../../../../../..//common/cuda-10.1/bin/..//include"   -m64 -o "kernel.o" "kernel.cudafe1.cpp" 
#$ gcc -D__CUDA_ARCH__=700 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  -O3 "-I../../../../../../..//common/cuda-10.1/bin/..//include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=168 -include "cuda_runtime.h" -m64 "pagerank.cu" > "pagerank.cpp1.ii" 
#$ cicc --c++14 --gnu_version=70500 --allow_managed   -arch compute_70 -m64 -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "pagerank.fatbin.c" -tused -nvvmir-library "../../../../../../..//common/cuda-10.1/bin/../nvvm/libdevice/libdevice.10.bc" --gen_module_id_file --module_id_file_name "pagerank.module_id" --orig_src_file_name "pagerank.cu" --gen_c_file_name "pagerank.cudafe1.c" --stub_file_name "pagerank.cudafe1.stub.c" --gen_device_file_name "pagerank.cudafe1.gpu"  "pagerank.cpp1.ii" -o "pagerank.ptx"
#$ ptxas -arch=sm_70 -m64  "pagerank.ptx"  -o "pagerank.sm_70.cubin" 
#$ fatbinary --create="pagerank.fatbin" -64 "--image=profile=sm_70,file=pagerank.sm_70.cubin" "--image=profile=compute_70,file=pagerank.ptx" --embedded-fatbin="pagerank.fatbin.c" 
#$ gcc -E -x c++ -D__CUDACC__ -D__NVCC__  -O3 "-I../../../../../../..//common/cuda-10.1/bin/..//include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=168 -include "cuda_runtime.h" -m64 "pagerank.cu" > "pagerank.cpp4.ii" 
#$ cudafe++ --c++14 --gnu_version=70500 --allow_managed  --m64 --parse_templates --gen_c_file_name "pagerank.cudafe1.cpp" --stub_file_name "pagerank.cudafe1.stub.c" --module_id_file_name "pagerank.module_id" "pagerank.cpp4.ii" 
#$ gcc -D__CUDA_ARCH__=700 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -O3 "-I../../../../../../..//common/cuda-10.1/bin/..//include"   -m64 -o "pagerank.o" "pagerank.cudafe1.cpp" 
#$ nvlink --arch=sm_70 --register-link-binaries="PR_COAL_dlink.reg.c"  -m64   "-L../../../../../../..//common/cuda-10.1/bin/..//lib64/stubs" "-L../../../../../../..//common/cuda-10.1/bin/..//lib64" -cpu-arch=X86_64 "kernel.o" "pagerank.o" "parse.o" "util.o"  -lcudadevrt  -o "PR_COAL_dlink.sm_70.cubin"
#$ fatbinary --create="PR_COAL_dlink.fatbin" -64 -link "--image=profile=sm_70,file=PR_COAL_dlink.sm_70.cubin" --embedded-fatbin="PR_COAL_dlink.fatbin.c" 
#$ gcc -c -x c++ -DFATBINFILE="\"PR_COAL_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"PR_COAL_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  -O3 "-I../../../../../../..//common/cuda-10.1/bin/..//include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=168 -m64 -o "PR_COAL_dlink.o" "../../../../../../..//common/cuda-10.1/bin/crt/link.stub" 
#$ g++ -O3 -m64 -o "../../../../bin/10.1/release/PR_COAL" -Wl,--start-group "PR_COAL_dlink.o" "kernel.o" "pagerank.o" "parse.o" "util.o"   "-L../../../../../../..//common/cuda-10.1/bin/..//lib64/stubs" "-L../../../../../../..//common/cuda-10.1/bin/..//lib64" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group 
