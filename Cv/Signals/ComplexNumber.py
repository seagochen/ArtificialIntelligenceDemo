import math


class ComplexNumber(object):

    def __init__(self):
        self.real = 0
        self.imag = 0

    def modulated(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)
