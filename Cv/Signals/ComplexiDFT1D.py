from math import cos, sin

from Component import *
from ComplexDFT1D import dft_1d
from ComplexNumber import ComplexNumber


def generate_original_signals(pts: int):
    x_axis = np.linspace(-np.pi, np.pi, pts)  # 假设 1s 内完成一个时间周期，即 1Hz
    signal = gen_sin_freq(x_axis, 4) + gen_sin_freq(x_axis, 8, np.pi / 4)
    return x_axis, signal


def complex_multi(a, b, c, d):
    real = a * c - b * d
    imag = a * d + b * c
    return real, imag


def calculate_base_fx(Ck, freq, N):
    pi = np.pi  # pi
    k = freq  # sampling frequency
    sig_cos = []
    sig_sin = []

    for n in range(N):  # x(0), x(1), x(2), ... x(N)
        # real sig
        r_sig = cos(2 * pi * n * k / N)
        i_sig = sin(2 * pi * n * k / N)

        # real
        real, imag = complex_multi(r_sig, i_sig, Ck.real, Ck.imag)

        # reconstruct signals
        sig_cos.append(real)
        sig_sin.append(imag)

    return sig_cos, sig_sin


def sum_signals(final, sig):
    if len(final) <= 0:
        return sig

    for i in range(len(final)):
        final[i].real = sig[i].real + final[i].real
        final[i].imag = sig[i].imag + final[i].imag

    return final


def convert_to_np(final):
    cos = []
    sin = []

    for i in range(len(final)):
        cos[i].append(final[i].real)
        sin[i].append(final[i].real)

    return cos, sin


def idft_1d(dft):
    N = len(dft)
    freq_real, freq_imag = np.zeros(N), np.zeros(N)
    maximum_sampling_freq = N  # 希望使用的采样最大频率
    increased_sampling_freq = maximum_sampling_freq / N  # 采样率步进
    current_sampling_freq = 0  # 当前的采样率，从 0Hz 开始进行采样

    for ck in dft:
        fxc, fxs = calculate_base_fx(ck, current_sampling_freq, N)

        # sigma all basis frequencies
        freq_real = freq_real + np.array(fxc)
        freq_imag = freq_imag + np.array(fxs)

        # increase freq
        current_sampling_freq = current_sampling_freq + increased_sampling_freq

    return freq_real / N, freq_imag / N


def reconstruct(real, imag):
    signal = []
    for i in range(len(real)):
        # compute amplitude
        amp = np.sqrt(real[i] ** 2 + imag[i] ** 2)

        if real[i] < 0:
            signal.append(-amp)
        else:
            signal.append(amp)

    return signal


if __name__ == "__main__":
    _axis, _signal = generate_original_signals(50)
    _dft = dft_1d(_signal)

    # validate dft
    validate_dft(_signal, _dft)
    # show_curve(_axis, _signal)

    _real, _imag = idft_1d(_dft)
    _re_sig = reconstruct(_real, _imag)
    # show_curve(_axis, _re_sig)
    validate_idft(_signal, _re_sig)

