#!/usr/bin/env python


import os
import sys
import search






run_dir = sys.argv[1]
if not os.path.exists(run_dir):
    print (run_dir, "not exists")
    exit()


print(run_dir)




traces = search.search(run_dir)


