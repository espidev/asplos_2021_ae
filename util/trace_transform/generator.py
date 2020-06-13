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
                  help="Add load after special instruction", default=False)
(options, args) = parser.parse_args()

common.load_defined_yamls()

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))

for bench in benchmarks:
    edir, ddir, exe, argslist = bench
    for args in argslist:
        run_dir = os.path.join( options.trace_dir, exe, common.get_argfoldername( args ), "traces")
        if not os.path.exists(run_dir):
            print (run_dir, "not exists")
            exit()
        cfgs = set(options.config.split(","))
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
                    print line[:-1]
        flist.close()

        if "dump" in cfgs:
            filename = "dump.log"
            dump.dump(run_dir, vfunc_inst_lists, filename)

        if "eliminate" in cfgs:
            output_dir = os.path.join(options.output, exe, common.get_argfoldername(args), "traces")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print output_dir
            eliminate.eliminate(run_dir, vfunc_inst_lists, output_dir)

        if "replace" in cfgs:
            output_dir = os.path.join(options.output, exe, common.get_argfoldername(args), "traces")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print output_dir
            replace.replace(run_dir, vfunc_inst_lists, output_dir, options)

        if "copy_list" in cfgs:
            output_dir = os.path.join(options.output, exe, common.get_argfoldername(args), "traces")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            new_file = os.path.join(output_dir, "kernelslist.g")
            if os.path.isfile(new_file):
                os.remove(new_file)
            shutil.copyfile(listname, new_file)
