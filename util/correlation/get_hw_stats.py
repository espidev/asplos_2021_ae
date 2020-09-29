#!/usr/bin/env python

from optparse import OptionParser
import os
import subprocess
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
import sys
sys.path.insert(0,os.path.join(this_directory,"..","job_launching"))
import common
import re
import shutil
import glob
import datetime
import yaml

parser = OptionParser()
parser.add_option("-B", "--benchmark_list", dest="benchmark_list",
                 help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for " +\
                       "the benchmark suite names.",
                 default="rodinia_2.0-ft")
parser.add_option("-D", "--device_num", dest="device_num",
                 help="CUDA device number",
                 default="0")
parser.add_option("-m", "--metrics", dest="metrics",
                 help="nsight metrics to find",
                 default="Kernel Name,l1tex__t_sector_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__lsu_writeback_active.sum,l1tex__tex_writeback_active.sum")

(options, args) = parser.parse_args()
common.load_defined_yamls()

metrics_set = set(options.metrics.split(","))
#print(metrics_set)

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))
cuda_version = common.get_cuda_version( this_directory )
foutput_name = "hw_stats.csv"
foutput = open(foutput_name, "w")

for bench in benchmarks:
    edir, ddir, exe, argslist = bench
    specific_ddir = os.path.join(this_directory,ddir)
    for args in argslist:
        run_name = os.path.join( exe, common.get_argfoldername( args ) )
        this_run_dir = os.path.join(this_directory, "..", "..", "run_hw", "device-" + options.device_num, cuda_version, run_name)
        this_pattern = this_run_dir + '/*.csv.nsight'
        for fname in glob.glob(this_pattern):
            print(fname)
            foutput.write(fname)
            foutput.write("\n")
            flist = open(fname, "r")
            lines = flist.readlines()
            start = 0
            metrics_dict = dict()
            metric_line = []
            metrics_idx = []
            for line in lines:
                csv_line = line.split("\"")
                if start == 1 and csv_line[0] == "": # unit line
                    start = 2
                    for idx in metrics_idx:
                        if csv_line[idx] == "sector" or csv_line[idx] == "cycle":
                            metrics_dict[idx] = 0
                elif start == 0 and len(csv_line) > 10 and csv_line[1] == "ID": # metric line
                    start = 1
                    metric_line = csv_line
                    for idx,element in enumerate(csv_line):
                        if element in metrics_set:
                            metrics_idx.append(idx)
                elif start == 2: # print with metric line, unit line and data line
                    for idx in metrics_idx:
                        if idx in metrics_dict.keys():
                            metrics_dict[idx] += int(csv_line[idx].replace(",", ""))
                        #print(csv_line[idx])
                        foutput.write("\"")
                        foutput.write(csv_line[idx])
                        foutput.write("\"")
                        foutput.write(",")
                    foutput.write("\n")
            for idx,values in metrics_dict.items():
                print('\"{}\",\"{}\"'.format(metric_line[idx],values))
            flist.close()
foutput.close()

