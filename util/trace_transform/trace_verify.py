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
import serial_file

# define parameters
parser = OptionParser()
parser.add_option("-B", "--benchmark_list", dest="benchmark_list",
                 help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for " +\
                       "the benchmark suite names.",
                 default="oop")
parser.add_option("-D", "--device_num", dest="device_num",
                 help="CUDA device number",
                 default="0")
parser.add_option("-f", "--filter_kernels", dest="filter_kernels", default="",
                  help="A regex string that will filter the kernel names to correlate. " +\
                  "This is especially useful when you skip some kernels in simulation.")
(options, args) = parser.parse_args()

common.load_defined_yamls()

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))

outname = "verify.log"
outfile = open(outname, "w+")

cuda_version = common.get_cuda_version( this_directory )
for bench in benchmarks:
    edir, ddir, exe, argslist = bench
    print(exe)
    outfile.write(exe)
    outfile.write("\n")
    for args in argslist:
        run_name = os.path.join( exe, common.get_argfoldername( args ) )

        trace_run_dir = os.path.join(this_directory, "..", "..", "run_hw", "traces", "device-" + options.device_num, cuda_version, run_name, "traces")
        if not os.path.exists(trace_run_dir):
            print (trace_run_dir, "not exists")
            exit()
        listname = os.path.join(trace_run_dir, "kernelslist.g")

        flist = open(listname, "r")
        lines = flist.readlines()
        pattern = '^kernel*'
        warp_inst = []
        for trace_file in lines:
            if re.search(pattern, trace_file):
                sf = serial_file.serial_file(trace_run_dir + '/' + trace_file[:-1], "r")
                line = sf.readline()
                print line[:-1].split(' ')
                if not re.search(options.filter_kernels, line[:-1].split(' ')[3]):
                    print(line[:-1].split(' ')[3])
                    continue
                warp_inst_count = 0
                line = sf.readline()
                while line:
                    if line[:-1].split(' ')[0] == 'insts':
                        warp_inst_count += float(line[:-1].split(' ')[2])
                    line = sf.readline()
                warp_inst.append(warp_inst_count)
                sf.close()
        print(warp_inst)
        for v in warp_inst:
            outfile.write(str(v))
        outfile.write("\n")

        hw_run_dir = os.path.join(this_directory, "..", "..", "run_hw", "device-" + options.device_num, cuda_version, run_name)
        if not os.path.exists(hw_run_dir):
            print (hw_run_dir, "not exists")
            exit()
        hwname = None
        for name in os.listdir(hw_run_dir):
            if re.match('^.*\.csv$', name):
                hwname = os.path.join(hw_run_dir, name)
        hw_file = open(hwname, "r")

        hw_warp_inst = []
        lines = hw_file.readlines()
        for line in lines:
            res = line[:-1].split(',')
            if len(res) >= 5 and re.search(options.filter_kernels, res[3]):
                hw_warp_inst.append(float(res[5]))
        print(hw_warp_inst)
        for v in hw_warp_inst:
            outfile.write(str(v))
        outfile.write("\n")

        warp_inst_diff = []
        if len(warp_inst) != len(hw_warp_inst):
            exit("ERROR - Hardware kernels are not equal to trace kernels!")
        for i in range(len(warp_inst)):
            warp_inst_diff.append((warp_inst[i] - hw_warp_inst[i]) / hw_warp_inst[i])
        print [format(i, ".2%") for i in warp_inst_diff]
        for v in warp_inst_diff:
            outfile.write("%s " % i)
        outfile.write("\n")
        hw_file.close()
        flist.close()
outfile.close()
