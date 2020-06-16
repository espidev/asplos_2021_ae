
import pattern
import serial_file

def extract_loads(vfunc_lines):
    obj_reg = vfunc_lines[0].split(' ')[10]
    offset = vfunc_lines[1].split(' ')[-2]
    obj_addr = vfunc_lines[0].split(' ')[11:-2]
    call = vfunc_lines[3].split(' ')
    line = call[0:6]
    line.append("1")
    line.append(obj_reg)
    line.append("LD.E.64")
    line.append("1 R1 8 1 0x0000000000fffc04 0 0")
    special_inst = ""
    for e in line:
        special_inst = special_inst + str(e) + " "
    return special_inst


def extract_vfunc_insts(vfunc_lines):
    obj_reg = vfunc_lines[0].split(' ')[10]
    offset = vfunc_lines[1].split(' ')[-2]
    obj_addr = vfunc_lines[0].split(' ')[11:-2]
    call = vfunc_lines[3].split(' ')
    line = call[0:6]
    line.append("0")
    line.append("COAL")
    line.append("1")
    line.append(obj_reg)
    for e in obj_addr:
        line.append(e)
    line.append(offset)
    special_inst = ""
    for e in line:
        special_inst = special_inst + str(e) + " "
    return special_inst

def replace(trace_dir, vfc, outpath, options):
    for (fname, vfc_traces) in vfc:
        vfc_traces.reset_iterator()
        sf = serial_file.serial_file(trace_dir + '/' + fname, "r")
        out = open(outpath + '/' + fname, "w+")
        line = sf.readline()
        vfunc_lines = [[], [], [], []]
        vfunc_lines_idx = 0
        while line:
            if line[:-1].split(' ')[0] == 'insts':
                if options.load:
                    line = "insts = ", str(vfc_traces.get_insts() + vfc_traces.get_vfc_number() * 2)
                else:
                    line = "insts = ", str(vfc_traces.get_insts() + vfc_traces.get_vfc_number())
                out.writelines(line)
                out.write("\n")
            else:
                if vfc_traces.compare_trace(sf.get_linenum()):
                    if vfunc_lines_idx == 3:
                        vfunc_lines[vfunc_lines_idx] = line
                        vfunc_lines_idx = 0
                        if options.load:
                            line = extract_loads(vfunc_lines)
                            out.writelines(line)
                            out.write("\n")
                        line = extract_vfunc_insts(vfunc_lines)
                        out.writelines(line)
                        out.write("\n")
                    else:
                        vfunc_lines[vfunc_lines_idx] = line
                        vfunc_lines_idx += 1
                else:
                    out.write(line)
            line = sf.readline()
        sf.close()
        out.close()
