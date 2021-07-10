def convolution():
    from Mathematics.Convolution.ConKernels import convolution_test
    convolution_test()


def dft_1d():
    from Mathematics.FFT1D.ComplexDFT1D import dft_1d_demo
    from Mathematics.FFT1D.ComplexiDFT1D import idft_1d_demo


    dft_1d_demo()
    idft_1d_demo()


def dft_2d():
    from Mathematics.FFT2D.FFT2Demo import dft_2d_demo
    dft_2d_demo()


if __name__ == "__main__":
    # convolution demos
    convolution()

    # FFT 1D
    dft_1d()

    # FFT 2D
    dft_2d()
