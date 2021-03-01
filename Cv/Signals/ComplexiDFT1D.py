import numpy as np
from math import cos, sin, sqrt
from Component import gen_sin_freq, gen_cos_freq, show_curve
from ComplexNumber import ComplexNumber
from ComplexDFT1D import dft_1d


def generate_original_signals(pts: int):
    x_axis = np.linspace(-np.pi, np.pi, pts)  # 假设 1s 内完成一个时间周期，即 1Hz
    signal = gen_sin_freq(x_axis, 4, 1, np.pi / 4) + gen_sin_freq(x_axis, 18, 6)
    return x_axis, signal


def idft_1d(x_axis, frequencies, sampling_freq):
    maximum_sampling_freq = sampling_freq  # 希望使用的采样最大频率
    increased_sampling_freq = maximum_sampling_freq / len(frequencies)  # 采样率步进
    current_sampling_freq = 0  # 当前的采样率，从 0Hz 开始进行采样

    sum_freq = np.zeros(x_axis.size)

    for ck in range(len(frequencies)):
        # e^x = cos(x) + jsin(x)
        basis_real = gen_cos_freq(x_axis, current_sampling_freq)
        basis_imag = gen_sin_freq(x_axis, current_sampling_freq)

        # ck = [c(r), jc(j)]
        cr = ck.real
        cj = ck.imag

        # final = ck * e^x

        # 递增频率
        current_sampling_freq = current_sampling_freq + increased_sampling_freq



if __name__ == "__main__":
    x_axis, signal = generate_original_signals(50)
    dft = dft_1d(signal, len(signal), len(signal))
