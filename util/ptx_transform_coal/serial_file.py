class serial_file:
    def __init__(self, filename, fopt):
        self.file = open(filename, fopt)
        self.linenum = -1

    def readline(self):
        self.linenum += 1
        line = self.file.readline()
        return line

    def get_linenum(self):
        return self.linenum

    def close(self):
        self.file.close()
