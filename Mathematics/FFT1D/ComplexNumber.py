import math


class ComplexNumber(object):

    def __init__(self, real=0., imag=0.):
        self.real = real
        self.imag = imag

    def modulated(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)


def generate_test_cx():
    d0 = ComplexNumber(879.000000, 0.000000)
    d1 = ComplexNumber(90.893436, 13.541617)
    d2 = ComplexNumber(-79.685020, 5.178500)
    d3 = ComplexNumber(105.981226, -28.670453)
    d4 = ComplexNumber(-112.202453, -67.242233)
    d5 = ComplexNumber(-6.000000, 6.928203)
    d6 = ComplexNumber(-40.481226, -21.662298)