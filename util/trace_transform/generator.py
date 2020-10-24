#!/usr/bin/env python

from optparse import OptionParser
import os
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
import sys
sys.path.insert(0,os.path.join(this_directory,"..","job_launching"))
import common
import re
import shutil
import yaml
import search
import dump
import eliminate
import replace
import uinst
import glob

# define parameters
parser = OptionParser()
parser.add_option("-B", "--benchmark_list", dest="benchmark_list",
        help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for " +\
                "the benchmark suite names.",
                default="oop")
parser.add_option("-T", "--trace", dest="trace_dir",
        help="trace directory",
        default=".")
parser.add_option("-O", "--output", dest="output",
        help="output directory",
        default=".")
parser.add_option("-C", "--config", dest="config", action="store",
        help="Configures to perform traces on", default="search")
parser.add_option("-L", "--load", dest="load", action="store_true",
        help="Add load before special instruction", default=False)
parser.add_option("-R", "--risc", dest="risc", action="store_true",
        help="Add typepointer with risc instruction only", default=False)
parser.add_option("-k", "--kernel", dest="kernel",
        help="kernel regex", default="kernel_ProducerCell_create_car|kernel_traffic_light_step|kernel_Car_step_prepare_path|kernel_Car_step_move|DeviceScanInitKernel|DeviceScanKernel|kernel_compact_initialize|kernel_compact_cars|kernel_compact_swap_pointers|candidate_prepare|alive_prepare|candidate_update|alive_update|kernel_AnchorPullNode_pull|kernel_Spring_compute_force|kernel_Node_move|kernel_NodeBase_initialize_bfs|kernel_NodeBase_bfs_visit|kernel_NodeBase_bfs_set_delete_flags|kernel_Spring_bfs_delete|alive_update|prepare|update|ConnectedComponent|BFS|PageRank|render|Body_compute_force|Body_update|Body_initialize_merge|Body_prepare_merge|Body_update_merge|Body_delete_merged|parallel_do")
parser.add_option("-V", "--vfc", dest="vfc", action="store_true",
        help="Calculate vfc BTW", default=False)
(options, args) = parser.parse_args()

common.load_defined_yamls()

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))

cfgs = set(options.config.split(","))

inst_type = {
                    #Load/Store Instructions
                    "LD" : 0,
                    #For now, we ignore constant loads, consider it as ALU_OP, TO DO
                    "LDC" : 0,
                    "LDG" : 0,
                    "LDL" : 0,
                    "LDS" : 0,
                    "ST" : 0,
                    "STG" : 0,
                    "STL" : 0,
                    "STS" : 0,
                    "MATCH" : 0,
                    "QSPC" : 0,
                    "ATOM" : 0,
                    "ATOMS" : 0,
                    "ATOMG" : 0,
                    "RED" : 0,
                    "CCTL" : 0,
                    "CCTLL" : 0,
                    "ERRBAR" : 0,
                    "MEMBAR" : 0,
                    "CCTLT" : 0,
                    # floating point 32 inst
                    "FADD" : 1,
                    "FADD32I" : 1,
                    "FCHK" : 1,
                    "FFMA32I" : 1,
                    "FFMA" : 1,
                    "FMNMX" : 1,
                    "FMUL" : 1,
                    "FMUL32I" : 1,
                    "FSEL" : 1,
                    "FSET" : 1,
                    "FSETP": 1,
                    "FSWZADD" : 1,
                    # SFU
                    "MUFU" : 1,
                    # Floating Point 16 Instructions
                    "HADD2" : 1,
                    "HADD2_32I" : 1,
                    "HFMA2" : 1,
                    "HFMA2_32I" : 1,
                    "HMUL2" : 1,
                    "HMUL2_32I" : 1,
                    "HSET2" : 1,
                    "HSETP2": 1,
                    #Double Point Instructions
                    "DADD" : 1,
                    "DFMA" : 1,
                    "DMUL" : 1,
            "DSETP" : 1,
                    #Integer Instructions
                    "BMSK" : 1,
                    "BREV" : 1,
                    "FLO" : 1,
                    "IABS" : 1,
                    "IADD" : 1,
                    "IADD3" : 1,
                    "IADD32I" : 1,
                    "IDP" : 1,
                    "IDP4A" : 1,
                    "IMAD" : 1,
                    "IMMA" : 1,
                    "IMNMX" : 1,
                    "IMUL" : 1,
                    "IMUL32I" : 1,
                    "ISCADD" : 1,
                    "ISCADD32I" : 1,
                    "ISETP" : 1,
                    "LEA" : 1,
                    "LOP" : 1,
                    "LOP3" : 1,
                    "LOP32I" : 1,
                    "POPC" : 1,
                    "SHF" : 1,
                    "SHR" : 1,
                    "VABSDIFF" : 1,
                    "VABSDIFF4" : 1,
                    #Conversion Instructions
                    "F2F" : 1,
                    "F2I" : 1,
                    "I2F" : 1,
                    "I2I" : 1,
                    "I2IP": 1,
                    "FRND" : 1,
                    #Movement Instructions
                    "MOV" : 1,
                    "MOV32I" : 1,
                    "PRMT" : 1,
                    "SEL" : 1,
                    "SGXT" : 1,
                    "SHFL" : 1,
                    #Predicate Instructions
                    "PLOP3" : 1,
                    "PSETP" : 1,
                    "P2R" : 1,
                    "R2P" : 1,
                    #Texture Instructions
                    #For now, we ignore texture loads, consider it as ALU_OP
                    "TEX" : 1,
                    "TLD" : 1,
                    "TLD4" : 1,
                    "TMML" : 1,
                    "TXD" : 1,
                    "TXQ" : 1,
                    "B2R" : 1,
                    "CS2R" : 1,
                    "CSMTEST" : 1,
                    "DEPBAR" : 1,
                    "GETLMEMBASE" : 1,
                    "LEPC" : 1,
                    "NOP" : 1,
                    "PMTRIG" : 1,
                    "R2B" : 1,
                    "S2R" : 1,
                    "SETCTAID" : 1,
                    "SETLMEMBASE" : 1,
                    "VOTE" : 1,
                    "VOTE_VTG" : 1,
                    #Control Instructions
                    "BMOV" : 2,
                    "BPT" : 2,
                    "BRA" : 2,
                    "BREAK" : 2,
                    "BRX" : 2,
                    "BSSY" : 2,
                    "BSYNC" : 2,
                    "CALL" : 2,
                    "EXIT" : 2,
                    "JMP" : 2,
                    "JMX" : 2,
                    "KILL" : 2,
                    "NANOSLEEP" : 2,
                    "RET" : 2,
                    "RPCMOV" : 2,
                    "RTT" : 2,
                    "WARPSYNC" : 2,
                    "YIELD" : 2,
                    "BAR" : 2
                    }

if "nvbit_inst" in cfgs:
    for bench in benchmarks:
        edir, ddir, exe, argslist = bench
        for args in argslist:
            run_dir = os.path.join( options.trace_dir, exe, common.get_argfoldername( args ))
            if not os.path.exists(run_dir):
                print(run_dir, "not exists")
                continue
                #exit()
            this_pattern = run_dir + '/*.csv'
            for fname in glob.glob(this_pattern):
                #print(fname)
                instfile = open(fname, "r")
                lines = instfile.readlines()
                isMatchedKernel = False
                kernelName = None
                totalInst = 0
                vfc = 0
                res = {0: 0, 1: 0, 2: 0, 3: 0}
                for line in lines:
                    token = line[:-1].split(' ')
                    if len(token) > 9 and token[0] == "kernel":
                        # Find kernels
                        if token[3] == "void":
                            kernelName = token[4].split('(')[0].split('<')[0].rsplit(':')[-1]
                        else:
                            kernelName = token[3].split('(')[0].split('<')[0].rsplit(':')[-1]
                        if re.match(options.kernel, kernelName):
                            isMatchedKernel = True
                        else:
                            isMatchedKernel = False
                        # Add total inst
                        if isMatchedKernel:
                            totalInst += int(token[-4][:-1])
                        #res = {0: 0, 1: 0, 2: 0}
                    elif len(token) > 4 and token[3] == "=":
                        #print(token[2], token[4])
                        if options.vfc and token[2] == "VFC":
                            if isMatchedKernel:
                                vfc += int(token[4])
                        elif token[2].split('.')[0] in inst_type:
                            if isMatchedKernel:
                                res[inst_type[token[2].split('.')[0]]] += int(token[4])
                        else:
                            print(token[2].split('.')[0], "not listed")
                            exit()
                if totalInst != res[0] + res[1] + res[2]:
                    print('Data Wrong {},{},{},{}'.format(totalInst,res[0],res[1],res[2]))
                print('{},{},{},{}'.format(exe,res[0],res[1],res[2]))
                if options.vfc:
                    print('{},{},{}'.format(vfc,totalInst,vfc*1000.0/totalInst))
                instfile.close()
                
            
    exit()

for bench in benchmarks:
    edir, ddir, exe, argslist = bench
    for args in argslist:
        run_dir = os.path.join( options.trace_dir, exe, common.get_argfoldername( args ), "traces")
        if not os.path.exists(run_dir):
            print (run_dir, "not exists")
            exit()
        listname=os.path.join(run_dir, "kernelslist.g")

        flist = open(listname, "r")
        lines = flist.readlines()
        vfunc_inst_lists = []

        if "search" in cfgs:
            pattern = '^kernel*'
            for line in lines:
                if re.search(pattern, line):
                    traces = search.search(run_dir, line[:-1])
                    vfunc_inst_lists.append([line[:-1], traces])
                    print(line[:-1])

        if "search_call" in cfgs:
            pattern = '^kernel*'
            for line in lines:
                if re.search(pattern, line):
                    traces = search.search_call(run_dir, line[:-1])
                    vfunc_inst_lists.append([line[:-1], traces])
                    print(line[:-1])

        if "dump" in cfgs:
            filename = "dump.log"
            dump.dump(run_dir, vfunc_inst_lists, filename)

        if "eliminate" in cfgs:
            output_dir = os.path.join(options.output, exe, common.get_argfoldername(args), "traces")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print(output_dir)
            eliminate.eliminate(run_dir, vfunc_inst_lists, output_dir)

        if "replace" in cfgs:
            output_dir = os.path.join(options.output, exe, common.get_argfoldername(args), "traces")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print(output_dir)
            replace.replace(run_dir, vfunc_inst_lists, output_dir, options)

        if "uinst" in cfgs:
            # count instructions
            inst_dict_lists = []
            inst_dict = dict()
            pattern = '^kernel*'
            for line in lines:
                if re.search(pattern, line):
                    inst_dict = uinst.uinst(run_dir, line[:-1])
                    inst_dict_lists.append([line[:-1], inst_dict])
                    #print line[:-1]

            #analyze
            res = {
                    0: 0,
                    1: 0,
                    2: 0
                    }
            for filename, inst_dict in inst_dict_lists:
                for key in inst_dict:
                    if key in inst_type:
                        res[inst_type[key]] += inst_dict[key]
                    else:
                        print(key, "not listed")
                        exit()
            print ('{},{},{},{}'.format(exe,res[0],res[1],res[2]))

        if "copy_list" in cfgs:
            output_dir = os.path.join(options.output, exe, common.get_argfoldername(args), "traces")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            new_file = os.path.join(output_dir, "kernelslist.g")
            if os.path.isfile(new_file):
                os.remove(new_file)
            shutil.copyfile(listname, new_file)

        flist.close()
