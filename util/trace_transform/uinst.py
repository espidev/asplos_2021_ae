from serial_file import serial_file

def uinst(path, fname):
    # print path, fname

    inst_dict = dict()
    f = open(path + '/' + fname, "r")
    sf = serial_file(path + '/' + fname, "r")
    line = sf.readline()
    while line:
        line = line.split(' ')
        if len(line) > 7:
            if line[6] == '0':
                # 0 dst operand
                inst = line[7].split('.')[0]
                # print inst
                if inst in inst_dict:
                    inst_dict[inst] += 1
                else:
                    inst_dict[inst] = 0
            elif line[6] == '1':
                # 1 dst operand
                inst = line[8].split('.')[0]
                # print inst
                if inst in inst_dict:
                    inst_dict[inst] += 1
                else:
                    inst_dict[inst] = 0
        line = sf.readline()

    #print inst_dict
    sf.close()
    return inst_dict
