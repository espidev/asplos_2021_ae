class trace:
    def __init__(self, insts):
        self.insts = insts
        self.func_list = []

class traces:
    def __init__(self):
        self.trace_list = []
        self.trace_idx = -1
        self.func_idx = -1
        self.line_idx = -1

    def append(self, trace):
        self.trace_list.append(trace)

    def tail(self):
        if len(self.trace_list):
            return self.trace_list[-1]
        else:
            print "No tail in traces"
            exit()

    def reset_iterator(self):
        self.trace_idx = -1
        self.func_idx = -1
        self.line_idx = -1

    def get_insts(self):
        if self.trace_idx + 1 < len(self.trace_list):
            self.trace_idx += 1
            self.func_idx = -1
            self.line_idx = -1
            return self.trace_list[self.trace_idx].insts
        else:
            print "No next trace in traces"
            exit()

    def has_next(self):
        if self.func_idx + 1 < len(self.trace_list[self.trace_idx].func_list):
            return True
        else:
            if self.func_idx >= 0 and self.line_idx + 1 < len(self.trace_list[self.trace_idx].func_list[self.func_idx]):
                return True
            else:
                return False

    def next(self):
        if self.func_idx >= 0 and self.line_idx + 1 < len(self.trace_list[self.trace_idx].func_list[self.func_idx]):
            self.line_idx += 1
        else:
            if self.func_idx + 1 < len(self.trace_list[self.trace_idx].func_list):
                self.func_idx += 1
                self.line_idx = 0
            else:
                print "next() does not exist"
                exit()

    def get_next(self):
        if self.func_idx >= 0 and self.line_idx + 1 < len(self.trace_list[self.trace_idx].func_list[self.func_idx]):
            return self.trace_list[self.trace_idx].func_list[self.func_idx][self.line_idx + 1]
        else:
            if self.func_idx + 1 < len(self.trace_list[self.trace_idx].func_list):
                return self.trace_list[self.trace_idx].func_list[self.func_idx + 1][0]
            else:
                print "get_next() does not exist"

    def compare_trace(self, num):
        if self.has_next():
            line_num = self.get_next()
            if line_num == num:
                self.next()
                return True
            else:
                return False
        else:
            return False

