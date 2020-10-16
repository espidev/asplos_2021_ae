#!/usr/bin/env python

from pattern import pattern
from trace import trace
from trace import traces
from serial_file import serial_file
from ring_buffer import ring_buffer

def search(path, fname):
    print path, fname
    # create pattern
    ptrns = []
    ptrn = pattern(8, "CALL.REL.NOINC", 7, "R[0-9]*")
    ptrns.append(ptrn)
    ptrn = pattern(8, "LDC.64", 7, "R[0-9]*")
    ptrns.append(ptrn)
    ptrn = pattern(8, "LD.E.SYS", 10, "R[0-9]*")
    ptrns.append(ptrn)
    ptrn = pattern(8, "LD.E.64.SYS", 10, "R[0-9]*")
    ptrns.append(ptrn)
    # create buffer
    buf = ring_buffer(100)
    #search
    res = traces()
    f = open(path + '/' + fname, "r")
    sf = serial_file(path + '/' + fname, "r")
    line = sf.readline()
    while line:
        buf.append(line)
        if line[:-1].split(' ')[0] == 'insts':
            vf_trace = trace(int(line.split(' ')[2]))
            res.append(vf_trace)
        if ptrns[0].match(line):
            pattern_idx = 1
            inst = [sf.get_linenum()]
            while pattern_idx < len(ptrns):
                offset = buf.find(ptrns[pattern_idx])
                if offset != -1:
                    inst.append(sf.get_linenum() - offset)
                    pattern_idx += 1
                else:
                    break
            if pattern_idx == len(ptrns):
                inst.reverse()
                res.tail().func_list.append(inst)
                res.tail().insts -= len(inst)

            else:
                print "Cannot find pattern in buf at line", sf.get_linenum()
                exit()
        line = sf.readline()

    sf.close()
    return res

def search_call(path, fname):
    print path, fname
    # create pattern
    ptrns = []
    ptrn = pattern(8, "CALL.REL.NOINC", 7, "R[0-9]*")
    ptrns.append(ptrn)
    ptrn = pattern(8, "LOP3.LUT", 7, "R[0-9]*")
    ptrns.append(ptrn)
    # create buffer
    buf = ring_buffer(100)
    #search
    res = traces()
    f = open(path + '/' + fname, "r")
    sf = serial_file(path + '/' + fname, "r")
    line = sf.readline()
    while line:
        buf.append(line)
        if line[:-1].split(' ')[0] == 'insts':
            vf_trace = trace(int(line.split(' ')[2]))
            res.append(vf_trace)
        if ptrns[0].match(line):
            inst = []
            offset = buf.find(ptrns[1])
            if offset != -1:
                inst = [sf.get_linenum() - offset]
                res.tail().func_list.append(inst)
                res.tail().insts -= len(inst)
            else:
                print "Warning: Cannot find pattern LOP.LUT in buf at line", sf.get_linenum()
        line = sf.readline()

    sf.close()
    return res