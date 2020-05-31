
#!/usr/bin/env python


from serial_file import serial_file

import re
import  sys

def clean_line(line):
    trace = re.split('\t| ', line)
    while ("" in trace):
        trace.remove("")
    return trace

def gen(fname):
   # create buffer
    buf = []
    f = open( 'coal.h' , "w")
    sf = serial_file(fname, "r")
    line = sf.readline()
    co = 0;
    vfun = "{   vtable = get_vfunc(ptr, range_tree, tree_size);  "
    class_name = ""
    while line:
        if "class" in line:
            co = 0;
            class_name = clean_line(line)[1];


        if "virtual" in line:

            str_name= clean_line(line);
            fun_name = str_name[str_name.index("virtual")+2]
            print(fun_name)
            fun_name = fun_name.replace('(',' ');
            fun_name = clean_line(fun_name)[0]
            def_coal = "#define COAL_" + class_name +"_"+fun_name+ "(ptr)"

            f.write(def_coal)
            f.write(vfun)
            tmo = "temp_coal = vtable["+str(co)+"]; }\n"
            f.write(tmo)

            co += 1
        line = sf.readline()

run_dir = sys.argv[1];
gen(run_dir)
