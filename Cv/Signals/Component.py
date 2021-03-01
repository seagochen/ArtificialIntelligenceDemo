import numpy as np
from matplotlib import pyplot as plt


def gen_sin_freq(x_axis, hz, amplitude=1, phase=0):
    return amplitude * np.sin(x_axis * hz + phase)


def gen_cos_freq(x_axis, hz, amplitude=1, phase=0):
    return amplitude * np.cos(x_axis * hz + phase)


def show_bars(data):
    plt.bar(range(len(data)), data)
    plt.title('fft freq chart')
    plt.show()


def show_curve(x, curve):
    plt.plot(x, curve)
    plt.show()


def validate_dft(signal, my_dft):
    np_dft = np.fft.fft(signal, len(signal))

    for i in range(50):
        np_real = round(np_dft[i].real, 3)
        np_imag = round(np_dft[i].imag, 3)
        my_real = round(my_dft[i].real, 3)
        my_imag = round(my_dft[i].img, 3)

        var_real = np_real ** 2 - my_real ** 2
        var_imag = np_imag ** 2 - my_imag ** 2

        print(f"np:\t{np_real}\t{np_imag}\t\tdft:\t{my_real}\t{my_imag}\t\tvar\t{var_real}\t{var_imag}")


def validate_idft(xaxis, signal):
    np_dft = np.fft.fft(signal, len(signal))
    np_sig = np.fft.ifft(np_dft)

    show_curve(xaxis, np_sig)