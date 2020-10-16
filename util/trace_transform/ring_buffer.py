from pattern import pattern

class ring_buffer:
    def __init__(self, size):
        self.buf = [None] * size
        self.head = 0
        self.size = size

    def append(self, line):
        self.buf[self.head] = line
        self.head = (self.head + 1) % self.size

    def find(self, pattern):
        index = (self.head + self.size - 1) % self.size
        while index != self.head:
            if pattern.match(self.buf[index]):
                return (self.head + self.size - index - 1) % self.size
            index = (index + self.size - 1) % self.size
        print "Cannot find pattern in the buffer"
        return -1
