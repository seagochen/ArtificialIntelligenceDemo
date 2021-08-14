from math import cos, sin
from Mathematics.FFT1D.Component import *
from Mathematics.FFT1D.ComplexNumber import ComplexNumber


def generate_original_signals(pts: int):
    x_axis = np.linspace(-np.pi, np.pi, pts)  # 假设 1s 内完成一个时间周期，即 1Hz
    signal = gen_sin_freq(x_axis, 4, 1, np.pi / 4) + gen_sin_freq(x_axis, 18, 6)
    return x_axis, signal


def calculate_ck(signals, N, k):
    C_k = ComplexNumber()  # ck以复数形式进行表示
    pi = np.pi  # pi
    n = 0  # 采样点
    steps = round(len(signals) / N)  # 采样步进，默认的采样步进为1，采样方式为全尺寸采样

    while n < N:
        # 这里的 k 是采样频率
        C_k.real = C_k.real + cos(2 * pi * n * k / N) * signals[steps * n]
        C_k.imag = C_k.imag - sin(2 * pi * n * k / N) * signals[steps * n]

        n += 1

    return C_k


def dft_1d(signals):
    dft_bins = []

    maximum_sampling_freq = len(signals)  # 希望使用的采样最大频率
    increased_sampling_freq = maximum_sampling_freq / len(signals)  # 采样率步进
    current_sampling_freq = 0  # 当前的采样率，从 0Hz 开始进行采样

    while current_sampling_freq < maximum_sampling_freq:
        ck = calculate_ck(signals, len(signals), current_sampling_freq)
        dft_bins.append(ck)

        current_sampling_freq += increased_sampling_freq

    return dft_bins


def dft_1d_demo():
    xs, sigs = generate_original_signals(100)
    dft = dft_1d(sigs)
    validate_dft(sigs, dft)