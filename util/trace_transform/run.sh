#./generator.py -T /scratch/tgrogers-disk01/a/zhan2308/oo/trace_buffer/baseline/ -C search,eliminate,copy_list -O /scratch/tgrogers-disk01/a/zhan2308/trace_mod
#./generator.py -T /scratch/tgrogers-disk01/a/zhan2308/oo/trace_buffer/baseline/ -C search,replace,copy_list -O /scratch/tgrogers-disk01/a/zhan2308/trace_mod
#./generator.py -T /scratch/tgrogers-disk01/a/zhan2308/oo/trace_buffer/OO_MEM/ -C search,eliminate,copy_list -B oop_mem -O /scratch/tgrogers-disk01/a/zhan2308/oo/trace_buffer/fix_OO_LIM/
./generator.py -T /scratch/tgrogers-disk01/a/zhan2308/oo/trace_buffer/OO_MEM/ -C search,replace,copy_list -B oop_mem -O /scratch/tgrogers-disk01/a/zhan2308/oo/trace_buffer/fix_OO_COAL_LD/
