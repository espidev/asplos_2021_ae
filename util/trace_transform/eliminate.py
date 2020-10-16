from serial_file import serial_file

def eliminate(path, vfc, outpath):
    for (fname, vfc_traces) in vfc:
        vfc_traces.reset_iterator()
        sf = serial_file(path + '/' + fname, "r")
        out = open(outpath + '/' + fname, "w+")
        line = sf.readline()
        while line:
            if line[:-1].split(' ')[0] == "insts":
                line = "insts = ", str(vfc_traces.get_insts())
                out.writelines(line)
                out.write("\n")
            elif not vfc_traces.compare_trace(sf.get_linenum()):
                out.write(line)

            line = sf.readline()
        sf.close()
        out.close()
