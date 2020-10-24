import re

class pattern:
    def __init__(self, op_pos, op, oprand_pos, oprand):
        self.op = op
        self.op_pos = op_pos
        self.oprand = oprand
        self.oprand_pos = oprand_pos

    def match_op(self, line):
        trace = line.split(' ')
        return len(trace) > self.op_pos and re.match(self.op, trace[self.op_pos])

    def match_oprand(self, line):
        trace = line.split(' ')
        return len(trace) > self.oprand_pos and re.match(self.oprand, trace[self.oprand_pos])

    def match(self, line):
        return self.match_op(line) and self.match_oprand(line)

    def retrieve_oprand(self, line):
        trace = line.split(' ')
        if self.match_op(line):
            return trace[self.oprand_pos]
        else:
            print("Fail to retrieve oprand ", line)
            return -1

    def retrieve_address(self, line):
        trace = line.split(' ')
        if self.match_op(line):
            mask=bin(int(trace[5], 16))
            bits= [ones for ones in mask[2:] if ones == '1']
            end = len(bits) + 13
            if trace[12] == '0':
                return trace[13:end]
        else:
            print("Fail to retrieve oprand ", line)
            return []

    def retrieve_offset(self, line):
        trace = line.split(' ')
        if self.match_op(line):
            return trace[-2]
        else:
            print("Fail to retrieve offset ", line)
            return -1
