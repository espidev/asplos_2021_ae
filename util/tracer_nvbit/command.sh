# #./run_hw_trace.py -B oop -D 0



(./run_hw_trace.py -B dynasoar:collision_MEM -D 0 ;./run_hw_trace.py -B dynasoar:game-of-life -D 0 ;./run_hw_trace.py -B dynasoar:structureV_COAL -D 0 ;) &
(./run_hw_trace.py -B dynasoar:generationV_MEM -D 1 ;./run_hw_trace.py -B dynasoar:collision -D 1 ;./run_hw_trace.py -B dynasoar:trafficV_COAL -D 1 ;) &
(./run_hw_trace.py -B dynasoar:trafficV_MEM -D 2  ;./run_hw_trace.py -B dynasoar:generationV_CONCORD -D 2 ;) &
(./run_hw_trace.py -B dynasoar:structureV_MEM -D 3 ;./run_hw_trace.py -B dynasoar:structureV_CONCORD -D 3  ;./run_hw_trace.py -B dynasoar:game-of-life_COAL -D 3 ;) &
(./run_hw_trace.py -B dynasoar:game-of-life_MEM -D 4 ;./run_hw_trace.py -B dynasoar:trafficV_CONCORD -D 4 ;./run_hw_trace.py -B dynasoar:collision_COAL -D 4 ;) &
(./run_hw_trace.py -B dynasoar:generationV -D 5  ; ./run_hw_trace.py -B dynasoar:game-of-life_CONCORD -D 5 ;) &
(./run_hw_trace.py -B dynasoar:structureV -D 6  ;  ./run_hw_trace.py -B dynasoar:collision_CONCORD -D 6 ;) &
(./run_hw_trace.py -B dynasoar:trafficV -D 7 ; ./run_hw_trace.py -B dynasoar:generationV_COAL -D 7 ;) &


(./run_hw_trace.py -B oop_trace:PR  -D 0;       ./run_hw_trace.py -B oop_coal:PR_COAL  -D 0;./run_hw_trace.py -B oop_mem:PR_MEM  -D 0 ;)&
(./run_hw_trace.py -B oop_trace:PRV  -D 1;      ./run_hw_trace.py -B oop_coal:BFS_COAL  -D 1;./run_hw_trace.py -B oop_mem:PRV_MEM  -D 1;)&
(./run_hw_trace.py -B oop_trace:CCV  -D 2;      ./run_hw_trace.py -B oop_concord:CCV_CONCORD  -D 2;./run_hw_trace.py -B oop_mem:CCV_MEM  -D 2;)&
(./run_hw_trace.py -B oop_trace:CC  -D 3;       ./run_hw_trace.py -B oop_concord:CC_CONCORD  -D 3;./run_hw_trace.py -B oop_mem:CC_MEM  -D 3;)&
(./run_hw_trace.py -B oop_trace:BFSV  -D 4;     ./run_hw_trace.py -B oop_concord:BFSV_CONCORD  -D 4;./run_hw_trace.py -B oop_mem:BFSV_MEM  -D 4;)&
(./run_hw_trace.py -B oop_trace:BFS  -D 5;      ./run_hw_trace.py -B oop_concord:BFS_CONCORD  -D 5;./run_hw_trace.py -B oop_mem:BFS_MEM  -D 5;)&
(./run_hw_trace.py -B oop_concord:PR_CONCORD  -D 6;./run_hw_trace.py -B oop_coal:PRV_COAL  -D 6;./run_hw_trace.py -B oop_coal:CCV_COAL  -D 6;)&
(./run_hw_trace.py -B oop_concord:PRV_CONCORD  -D 7;./run_hw_trace.py -B oop_coal:CC_COAL  -D 7;./run_hw_trace.py -B oop_coal:BFSV_COAL  -D 7;)&






export DYNAMIC_KERNEL_LIMIT_END=208;
export DYNAMIC_KERNEL_LIMIT_START=211;

./run_hw_trace.py -B dynasoar:game-of-life_CONCORD -D 5 &
export DYNAMIC_KERNEL_LIMIT_END=208;export DYNAMIC_KERNEL_LIMIT_START=205 ;./run_hw_trace.py -B dynasoar:game-of-life -D 0 &
export DYNAMIC_KERNEL_LIMIT_END=211;export DYNAMIC_KERNEL_LIMIT_START=208 ;./run_hw_trace.py -B dynasoar:game-of-life_COAL -D 3 &
./run_hw_trace.py -B dynasoar:game-of-life_MEM -D 4 &

./run_hw_trace.py -B dynasoar:nbodyV_MEM -D 4 &
./run_hw_trace.py -B dynasoar:nbodyV_CONCORD -D 5 &
./run_hw_trace.py -B dynasoar:nbodyV -D 0 &
./run_hw_trace.py -B dynasoar:nbodyV_COAL -D 3 &



./run_hw_trace.py -B dynasoar:game-of-life_MEM -D 4 &
./run_hw_trace.py -B dynasoar:game-of-life_CONCORD -D 5 &
./run_hw_trace.py -B dynasoar:game-of-life_COAL -D 3 &



./run_hw_trace.py -B oop_trace:RAY  -D 0 &
./run_hw_trace.py -B oop_concord:RAY_CONCORD  -D 6 &
./run_hw_trace.py -B oop_mem:RAY_MEM  -D 3 &
./run_hw_trace.py -B oop_coal:RAY_COAL  -D 7 &



./run_hw_trace.py -B oop_coal:CCV_COAL  -D 6 &
./run_hw_trace.py -B oop_coal:CC_COAL  -D 7 &




/////

export DYNAMIC_KERNEL_LIMIT_START=9081;export DYNAMIC_KERNEL_LIMIT_END=9084;

(export DYNAMIC_KERNEL_LIMIT_START=4;   export DYNAMIC_KERNEL_LIMIT_END=9;./run_hw_trace.py -B dynasoar:collision_MEM -D 0 ;     export DYNAMIC_KERNEL_LIMIT_START=205;export DYNAMIC_KERNEL_LIMIT_END=208; ./run_hw_trace.py -B dynasoar:game-of-life -D 0 ;       export DYNAMIC_KERNEL_LIMIT_START=22;export DYNAMIC_KERNEL_LIMIT_END=36;./run_hw_trace.py -B dynasoar:structureV_COAL -D 0 ;) &
(export DYNAMIC_KERNEL_LIMIT_START=307; export DYNAMIC_KERNEL_LIMIT_END=309;./run_hw_trace.py -B dynasoar:generationV_MEM -D 1 ; export DYNAMIC_KERNEL_LIMIT_START=2;export DYNAMIC_KERNEL_LIMIT_END=7;./run_hw_trace.py -B dynasoar:collision -D 1 ;               export DYNAMIC_KERNEL_LIMIT_START=9083;export DYNAMIC_KERNEL_LIMIT_END=9086;./run_hw_trace.py -B dynasoar:trafficV_COAL -D 1 ;) &
(export DYNAMIC_KERNEL_LIMIT_START=9079;export DYNAMIC_KERNEL_LIMIT_END=9082;./run_hw_trace.py -B dynasoar:trafficV_MEM -D 2  ;  export DYNAMIC_KERNEL_LIMIT_START=304;export DYNAMIC_KERNEL_LIMIT_END=306;./run_hw_trace.py -B dynasoar:generationV_CONCORD -D 2 ;) &
(export DYNAMIC_KERNEL_LIMIT_START=22;  export DYNAMIC_KERNEL_LIMIT_END=36;./run_hw_trace.py -B dynasoar:structureV_MEM -D 3 ;   export DYNAMIC_KERNEL_LIMIT_START=20;export DYNAMIC_KERNEL_LIMIT_END=34;./run_hw_trace.py -B dynasoar:structureV_CONCORD -D 3  ;   export DYNAMIC_KERNEL_LIMIT_START=208;export DYNAMIC_KERNEL_LIMIT_END=211;./run_hw_trace.py -B dynasoar:game-of-life_COAL -D 3 ;) &
(export DYNAMIC_KERNEL_LIMIT_START=208; export DYNAMIC_KERNEL_LIMIT_END=211;./run_hw_trace.py -B dynasoar:game-of-life_MEM -D 4; export DYNAMIC_KERNEL_LIMIT_START=9081;export DYNAMIC_KERNEL_LIMIT_END=9084;./run_hw_trace.py -B dynasoar:trafficV_CONCORD -D 4 ;  export DYNAMIC_KERNEL_LIMIT_START=4;export DYNAMIC_KERNEL_LIMIT_END=9;./run_hw_trace.py -B dynasoar:collision_COAL -D 4 ;) &
(export DYNAMIC_KERNEL_LIMIT_START=304; export DYNAMIC_KERNEL_LIMIT_END=306;./run_hw_trace.py -B dynasoar:generationV -D 5  ;    export DYNAMIC_KERNEL_LIMIT_START=205;export DYNAMIC_KERNEL_LIMIT_END=208;./run_hw_trace.py -B dynasoar:game-of-life_CONCORD -D 5 ;) &
(export DYNAMIC_KERNEL_LIMIT_START=20;  export DYNAMIC_KERNEL_LIMIT_END=34;./run_hw_trace.py -B dynasoar:structureV -D 6  ;      export DYNAMIC_KERNEL_LIMIT_START=2;export DYNAMIC_KERNEL_LIMIT_END=7; ./run_hw_trace.py -B dynasoar:collision_CONCORD -D 6 ;) &
(export DYNAMIC_KERNEL_LIMIT_START=9081;export DYNAMIC_KERNEL_LIMIT_END=9084;./run_hw_trace.py -B dynasoar:trafficV -D 7 ;       export DYNAMIC_KERNEL_LIMIT_START=307;export DYNAMIC_KERNEL_LIMIT_END=309;./run_hw_trace.py -B dynasoar:generationV_COAL -D 7 ;) &





(export DYNAMIC_KERNEL_LIMIT_START=208; export DYNAMIC_KERNEL_LIMIT_END=211;./run_hw_trace.py -B dynasoar:game-of-life_MEM -D 0;) &
(export DYNAMIC_KERNEL_LIMIT_START=205;export DYNAMIC_KERNEL_LIMIT_END=208; ./run_hw_trace.py -B dynasoar:game-of-life -D 1 ; ) &
(export DYNAMIC_KERNEL_LIMIT_START=205;export DYNAMIC_KERNEL_LIMIT_END=208;./run_hw_trace.py -B dynasoar:game-of-life_CONCORD -D 2 ;) &
(export DYNAMIC_KERNEL_LIMIT_START=208;export DYNAMIC_KERNEL_LIMIT_END=211;./run_hw_trace.py -B dynasoar:game-of-life_COAL -D 3 ;) &
(export DYNAMIC_KERNEL_LIMIT_START=208;export DYNAMIC_KERNEL_LIMIT_END=211;./run_hw_trace.py -B dynasoar:game-of-life_TP -D 5 ;) &

squeue -u aalawneh

./run_simulations.py -T ~/traces_selected_itr/dynasoar_concord/ -C QV100-TRACE -N DYNASOAR_CONCORD -B dynasoar_concord -M 50000
./run_simulations.py -T ~/traces_selected_itr/dynasoar/ -C QV100-TRACE -N DYNASOAR_CUDA -B dynasoar_cuda -M 50000
./run_simulations.py -T ~/traces_selected_itr/dynasoar_coal/ -C QV100-TRACE -N DYNASOAR_COAL -B dynasoar_coal -M 50000
./run_simulations.py -T ~/traces_selected_itr/dynasoar_mem/ -C QV100-TRACE -N DYNASOAR_MEM -B dynasoar_mem -M 50000


./run_simulations.py -T ~/traces_selected_itr/oop -C QV100-TRACE -N OOP_CUDA -B oop -M 50000
./run_simulations.py -T ~/traces_selected_itr/oop_concord/ -C QV100-TRACE -N OOP_CONCORD -B oop_concord -M 50000
./run_simulations.py -T ~/traces_selected_itr/oop_coal/ -C QV100-TRACE -N OOP_COAL -B oop_coal -M 50000
./run_simulations.py -T ~/traces_selected_itr/oop_mem/ -C QV100-TRACE -N OOP_MEM -B oop_mem -M 50000



### transform to type pointer no load
./generator.py -B dynasoar_mem -T /home/tgrogers-raid/a/aalawneh/traces_selected_itr/ -O /home/tgrogers-raid/a/aalawneh/type_pointer_no_load -C search,replace,copy_list &
./generator.py -B oop_mem -T /home/tgrogers-raid/a/aalawneh/traces_selected_itr/ -O /home/tgrogers-raid/a/aalawneh/type_pointer_no_load -C search,replace,copy_list &


### transform to type pointer load
./generator.py -B dynasoar_mem -T /home/tgrogers-raid/a/aalawneh/traces_selected_itr/ -O /home/tgrogers-raid/a/aalawneh/type_pointer_load -L -C search,replace,copy_list &
./generator.py -B oop_mem -T /home/tgrogers-raid/a/aalawneh/traces_selected_itr/ -O /home/tgrogers-raid/a/aalawneh/type_pointer_load -L -C search,replace,copy_list &



( export DYNAMIC_KERNEL_LIMIT_START=0;export DYNAMIC_KERNEL_LIMIT_END=306;./run_hw_trace.py -B oop -D 0 ; ) &
( export DYNAMIC_KERNEL_LIMIT_START=0;export DYNAMIC_KERNEL_LIMIT_END=306;./run_hw_trace.py -B oop_mem -D 1 ; ) &
( export DYNAMIC_KERNEL_LIMIT_START=0;export DYNAMIC_KERNEL_LIMIT_END=306;./run_hw_trace.py -B oop_concord -D 2 ; ) &
( export DYNAMIC_KERNEL_LIMIT_START=0;export DYNAMIC_KERNEL_LIMIT_END=306;./run_hw_trace.py -B oop_coal -D 3 ; ) &
( export DYNAMIC_KERNEL_LIMIT_START=0;export DYNAMIC_KERNEL_LIMIT_END=306;./run_hw_trace.py -B oop_tp -D 5 ; ) &





### run with 100
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N DYNASOAR_CONCORD_5 -B dynasoar_concord -M 256000
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N DYNASOAR_CUDA_5 -B dynasoar_cuda -M 256000
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N DYNASOAR_COAL_5 -B dynasoar_coal -M 256000
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N DYNASOAR_MEM_5 -B dynasoar_mem -M 256000
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N OOP_CUDA_5 -B oop -M 256000
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N OOP_CONCORD_5 -B oop_concord -M 256000
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N OOP_COAL_5 -B oop_coal -M 256000
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N OOP_MEM_5 -B oop_mem -M 256000


./run_simulations.py -T ~/type_pointer_no_load/ -C QV100-TRACE -N OOP_TP_NO -B oop_mem -M 256000
./run_simulations.py -T ~/type_pointer_no_load/ -C QV100-TRACE -N DYNASOAR_TP_NO -B dynasoar_mem -M 256000


./run_simulations.py -T ~/type_pointer_load/ -C QV100-TRACE -N OOP_TP_LOAD -B oop_mem -M 256000
./run_simulations.py -T ~/type_pointer_load/ -C QV100-TRACE -N DYNASOAR_TP_LOAD -B dynasoar_mem -M 256000

./job_status.py -N DYNASOAR_CONCORD_5
./job_status.py -N DYNASOAR_CUDA_5 
./job_status.py -N DYNASOAR_COAL_5 
./job_status.py -N DYNASOAR_MEM_5 
./job_status.py -N OOP_CUDA_5 
./job_status.py -N OOP_CONCORD_5 
./job_status.py -N OOP_COAL_5 
./job_status.py -N OOP_MEM_5 
./job_status.py -N OOP_TP_NO 
./job_status.py -N DYNASOAR_TP_NO
./job_status.py -N OOP_TP_LOAD
./job_status.py -N DYNASOAR_LOAD


./get_stats.py -N DYNASOAR_CONCORD_5 > results.csv
./get_stats.py -N DYNASOAR_CUDA_5  >>results.csv
./get_stats.py -N DYNASOAR_COAL_5 
./get_stats.py -N DYNASOAR_MEM_5 




./generator.py -B oop_mem -T /home/tgrogers-raid/a/aalawneh/traces_selected_itr/ -O /home/tgrogers-raid/a/aalawneh/type_pointer_no_load_BFS -C search,replace,copy_list &
./generator.py -B oop_mem -T /home/tgrogers-raid/a/aalawneh/traces_selected_itr/ -O /home/tgrogers-raid/a/aalawneh/type_pointer_load_BFS -L -C search,replace,copy_list &




./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N DYNASOAR_CONCORD_REMAIN -B dynasoar_concord -M 256000
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N DYNASOAR_CUDA_REMAIN -B dynasoar_cuda -M 256000
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N DYNASOAR_COAL_REMAIN -B dynasoar_coal -M 256000


./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N OOP_FIXED_2 -B oop  -M 100000
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N OOP_CONCORD_FIXED_2 -B oop_concord  -M 100000
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N OOP_COAL_FIXED_2 -B oop_coal  -M 100000
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N OOP_MEM_FIXED_2 -B oop_mem  -M 100000
./run_simulations.py -T ~/type_pointer_no_load/ -C QV100-TRACE -N OOP_TP_NO_FIXED_2 -B oop_tp_no   -M 100000
./run_simulations.py -T ~/type_pointer_load/ -C QV100-TRACE -N OOP_TP_LOAD_FIXED_2 -B oop_tp_load    -M 100000


./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N DYNASOAR_MEM_FIXED_2 -B dynasoar_mem -M 100000
./run_simulations.py -T ~/type_pointer_load/ -C QV100-TRACE -N DYNASOAR_TP_LOAD_FIXED_2 -B dynasoar_tp_load -M 100000
./run_simulations.py -T ~/type_pointer_no_load/ -C QV100-TRACE -N DYNASOAR_TP_NO_FIXED_2  -B dynasoar_tp_no -M 100000




./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N OOP_CUDA_5 -B oop -M 256000
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N OOP_CONCORD_5 -B oop_concord -M 256000
./run_simulations.py -T ~/traces_selected_itr/ -C QV100-TRACE -N OOP_COAL_5 -B oop_coal -M 256000




./generator.py -B oop -T /home/tgrogers-raid/a/aalawneh/traces_selected_itr/ -O /home/tgrogers-raid/a/aalawneh/type_pointer_cuda_tp_no -C search,replace,copy_list &
./generator.py -B oop -T /home/tgrogers-raid/a/aalawneh/traces_selected_itr/ -O /home/tgrogers-raid/a/aalawneh/type_pointer_cuda_tp_load -L -C search,replace,copy_list &


./generator.py -B dynasoar_cuda -T /home/tgrogers-raid/a/aalawneh/traces_selected_itr/ -O /home/tgrogers-raid/a/aalawneh/type_pointer_cuda_tp_no -C search,replace,copy_list &
./generator.py -B dynasoar_cuda -T /home/tgrogers-raid/a/aalawneh/traces_selected_itr/ -O /home/tgrogers-raid/a/aalawneh/type_pointer_cuda_tp_load -L -C search,replace,copy_list &

./run_simulations.py -T ~/type_pointer_cuda_tp_no/ -C QV100-TRACE -N CUDA_TP_NO -B oop_cuda_tp_no,oop_cuda_tp_load,dynasoar_cuda_tp_no,dynasoar_cuda_tp_load -M 256000

./run_simulations.py -T ~/type_pointer_cuda_tp_load/ -C QV100-TRACE -N CUDA_TP_LOAD -B oop_cuda_tp_load,dynasoar_cuda_tp_load -M 256000




(export DYNAMIC_KERNEL_LIMIT_START=43630;export DYNAMIC_KERNEL_LIMIT_END=43674;./run_hw_trace.py -B dynasoar:trafficV_MEM -D 0   ;export DYNAMIC_KERNEL_LIMIT_START=22;  export DYNAMIC_KERNEL_LIMIT_END=36;./run_hw_trace.py -B dynasoar:structureV_MEM -D 0 ;  export DYNAMIC_KERNEL_LIMIT_START=307;export DYNAMIC_KERNEL_LIMIT_END=309;./run_hw_trace.py -B dynasoar:generationV_COAL -D 0 ;   ) &
(export DYNAMIC_KERNEL_LIMIT_START=43634;export DYNAMIC_KERNEL_LIMIT_END=43678;./run_hw_trace.py -B dynasoar:trafficV_COAL -D 1   ;export DYNAMIC_KERNEL_LIMIT_START=20;export DYNAMIC_KERNEL_LIMIT_END=34;./run_hw_trace.py -B dynasoar:structureV_CONCORD -D 1  ; export DYNAMIC_KERNEL_LIMIT_START=307;export DYNAMIC_KERNEL_LIMIT_END=309;./run_hw_trace.py -B dynasoar:generationV_TP -D 1 ;) &
(export DYNAMIC_KERNEL_LIMIT_START=43623;export DYNAMIC_KERNEL_LIMIT_END=43667;./run_hw_trace.py -B dynasoar:trafficV_CONCORD -D 2   ; export DYNAMIC_KERNEL_LIMIT_START=22;export DYNAMIC_KERNEL_LIMIT_END=36;./run_hw_trace.py -B dynasoar:structureV_COAL -D 2 ;  export DYNAMIC_KERNEL_LIMIT_START=205;export DYNAMIC_KERNEL_LIMIT_END=208; ./run_hw_trace.py -B dynasoar:game-of-life -D 2 ;  ) &
(export DYNAMIC_KERNEL_LIMIT_START=43623;export DYNAMIC_KERNEL_LIMIT_END=43667;./run_hw_trace.py -B dynasoar:trafficV -D 3   ;export DYNAMIC_KERNEL_LIMIT_START=22;export DYNAMIC_KERNEL_LIMIT_END=36;./run_hw_trace.py -B dynasoar:structureV_TP -D 3 ; export DYNAMIC_KERNEL_LIMIT_START=205;export DYNAMIC_KERNEL_LIMIT_END=208;./run_hw_trace.py -B dynasoar:game-of-life_CONCORD -D 3 ;    ) &
(export DYNAMIC_KERNEL_LIMIT_START=43634;export DYNAMIC_KERNEL_LIMIT_END=43678;./run_hw_trace.py -B dynasoar:trafficV_TP -D 4   ; export DYNAMIC_KERNEL_LIMIT_START=208; export DYNAMIC_KERNEL_LIMIT_END=211;./run_hw_trace.py -B dynasoar:game-of-life_MEM -D 4;   ) &
(export DYNAMIC_KERNEL_LIMIT_START=307; export DYNAMIC_KERNEL_LIMIT_END=309;./run_hw_trace.py -B dynasoar:generationV_MEM -D 5 ; export DYNAMIC_KERNEL_LIMIT_START=208;export DYNAMIC_KERNEL_LIMIT_END=211;./run_hw_trace.py -B dynasoar:game-of-life_COAL -D 5 ;    ) &
(export DYNAMIC_KERNEL_LIMIT_START=304; export DYNAMIC_KERNEL_LIMIT_END=306;./run_hw_trace.py -B dynasoar:generationV -D 6  ;  export DYNAMIC_KERNEL_LIMIT_START=208;export DYNAMIC_KERNEL_LIMIT_END=211;./run_hw_trace.py -B dynasoar:game-of-life_TP -D 6 ;   ) &
(export DYNAMIC_KERNEL_LIMIT_START=304;export DYNAMIC_KERNEL_LIMIT_END=306;./run_hw_trace.py -B dynasoar:generationV_CONCORD -D 7 ;   export DYNAMIC_KERNEL_LIMIT_START=20;  export DYNAMIC_KERNEL_LIMIT_END=34;./run_hw_trace.py -B dynasoar:structureV -D 7  ; ) &





( export DYNAMIC_KERNEL_LIMIT_START=0;export DYNAMIC_KERNEL_LIMIT_END=306;./run_hw_trace.py -B oop -D 0 ; ) &
( export DYNAMIC_KERNEL_LIMIT_START=0;export DYNAMIC_KERNEL_LIMIT_END=306;./run_hw_trace.py -B oop_mem -D 1 ; ) &
( export DYNAMIC_KERNEL_LIMIT_START=0;export DYNAMIC_KERNEL_LIMIT_END=306;./run_hw_trace.py -B oop_concord -D 2 ; ) &
( export DYNAMIC_KERNEL_LIMIT_START=0;export DYNAMIC_KERNEL_LIMIT_END=306;./run_hw_trace.py -B oop_coal -D 3 ; ) &
( export DYNAMIC_KERNEL_LIMIT_START=0;export DYNAMIC_KERNEL_LIMIT_END=306;./run_hw_trace.py -B oop_tp -D 4 ; ) &
( export DYNAMIC_KERNEL_LIMIT_START=208;export DYNAMIC_KERNEL_LIMIT_END=211;./run_hw_trace.py -B dynasoar:game-of-life_TP -D 5 ;) &










./generator.py -B multiple_itr -T /home/tgrogers-raid/a/aalawneh/traces_multiple_kernels/ -O /home/tgrogers-raid/a/aalawneh/type_pointer_fixed_load_tp_load -L -C search,replace,copy_list &
./generator.py -B oop_mem -T /home/tgrogers-raid/a/aalawneh/traces_selected_itr/ -O /home/tgrogers-raid/a/aalawneh/type_pointer_fixed_load_tp_load -L -C search,replace,copy_list &
./generator.py -B dynasoar_mem -T /home/tgrogers-raid/a/aalawneh/traces_selected_itr/ -O /home/tgrogers-raid/a/aalawneh/type_pointer_fixed_load_tp_load -L -C search,replace,copy_list &





./run_simulations.py -T ~/traces_multiple_kernels/ -C QV100-TRACE -N BENCH_MULTI_ITR_2 -B multiple_itr 


./run_simulations.py -T ~/type_pointer_fixed_load_tp_load/ -C QV100-TRACE -N ALL_FIXED_TP_LOAD -B multiple_itr,dynasoar_tp_load,dynasoar_cuda_tp_load,oop_cuda_tp_load,oop_tp_load 
