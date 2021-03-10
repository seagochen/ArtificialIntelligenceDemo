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

    for i in range(len(signal)):
        np_real = round(np_dft[i].real, 4)
        np_imag = round(np_dft[i].imag, 4)
        my_real = round(my_dft[i].real, 4)
        my_imag = round(my_dft[i].imag, 4)

        var_real = (np_real - my_real) ** 2
        var_imag = (np_imag - my_imag) ** 2

        print(f"np:\t{np_real}\t{np_imag}\t\tdft:\t{my_real}\t{my_imag}\t\tvar\t{var_real}\t{var_imag}")


def validate_idft(signal, idft_sig):

    for i in range(len(signal)):
        sig = round(signal[i], 4)
        idf = round(idft_sig[i], 4)

        # var
        var = (sig - idf) ** 2

        print(f"pt:{i}\tsig:\t{sig}\t\tres:\t{idf}\t\tvar\t{var}")


def compute_modulate(dft):
    final = []
    for i in range(len(dft)):
        modulate = dft[i].real ** 2 + dft[i].imag ** 2
        modulate = np.sqrt(modulate)
        final.append(modulate)

    return final