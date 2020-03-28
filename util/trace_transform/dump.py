from trace import trace
from serial_file import serial_file

def dump(path, vfc, output):
    out = open(output, "w+")
    for (fname, vfc_traces) in vfc:
        print fname
        sf = serial_file(path + '/' + fname, "r")
        for vfc_trace in vfc_traces.trace_list:
            print vfc_trace.insts
            out.write(str(vfc_trace.insts))
            out.write("\n")
            print vfc_trace.func_list

            for lists in vfc_trace.func_list:
                for num in reversed(lists):
                    while sf.get_linenum() < num:
                        line = sf.readline()
                    out.writelines(line)
                out.write("\n")
        sf.close()
    out.close()

