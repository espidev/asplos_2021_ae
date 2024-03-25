#!/usr/bin/env python


from serial_file import serial_file

import re


def clean_line(line):
    trace = re.split('\t| ', line)
    while ("" in trace):
        trace.remove("")
    return trace
def remove_vfun(buf,to_replace):
    i = 0;
    to_replace = clean_line(to_replace)[0]
    for line in buf:
        if(to_replace in line):
            temp = clean_line(line);
            temp = temp[2].replace('[','')
            temp = temp.split('+')[0]
            buf[i]="\n"
            buf[i-1]="\n"
        i += 1


def search(fname):
    print  fname

    # create buffer
    buf = []
    #search


    f = open( fname+'_coal', "w")
    sf = serial_file( fname, "r")
    line = sf.readline()
    co = 0 ;
    while line:
        trace = clean_line(line)
        dest_reg=""

        buf = []

        if trace[0] == 'st.global.u64' and trace[1]=='[temp_coal],' :
            dest_reg = trace[2]
            curr_line = line
            to_replace= ""
            while True:
                temp = sf.readline()
                buf.append(temp)
                spl = clean_line(temp)
                if len(spl) >= 2 and spl[0] == 'st.global.u64' and spl[1] == '[temp_coal],':
                    ##error
                    f.write('%s' % line)
                    break
                elif len(spl) >= 2 and (spl[0] == 'call' or spl[0] == 'call.uni'):
                    temp = sf.readline()
                    to_replace=temp.replace(',','')
                    buf.append(dest_reg.replace(';',','))
                    remove_vfun(buf,to_replace)
                    co +=1
                    break
            for listitem in buf:
                f.write('%s' % listitem)




        else:
            f.write(line)
        line = sf.readline()

    sf.close()

