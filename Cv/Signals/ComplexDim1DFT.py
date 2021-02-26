import numpy as np
from math import cos, sin, sqrt
from matplotlib import pyplot as plt


class ComplexNumber(object):

    def __init__(self):
        self.real = 0
        self.img = 0

    def amplitude(self):
        return sqrt(self.real ** 2 + self.img ** 2)


def generate_original_signals(pts: int):
    x_axis = np.linspace(-np.pi, np.pi, pts)  # 假设 1s 内完成一个时间周期，即 1Hz
    return x_axis, np.sin(4 * x_axis + np.pi / 4) + 6 * np.sin(18 * x_axis)  # 根据 X 轴坐标生成 1s 内的原始信号


def calculate_ck(signals, N, k):
    C_k = ComplexNumber()  # ck以复数形式进行表示
    pi = np.pi  # pi
    n = 0  # 采样点
    steps = round(len(signals) / N)  # 采样步进，默认的采样步进为1，采样方式为全尺寸采样

    while n < N:
        # 这里的 k 是采样频率
        C_k.real = C_k.real + cos(2 * pi * n * k / N) * signals[steps * n]
        C_k.img = C_k.img - sin(2 * pi * n * k / N) * signals[steps * n]

        n += 1

    return C_k


def dft_1d(signals, sampling_freq, dft_lens):
    dft_bins = []

    maximum_sampling_freq = sampling_freq  # 希望使用的采样最大频率
    increased_sampling_freq = maximum_sampling_freq / dft_lens  # 采样率步进
    current_sampling_freq = 0  # 当前的采样率，从 0Hz 开始进行采样

    while current_sampling_freq < maximum_sampling_freq:
        ck = calculate_ck(signals, len(signals), current_sampling_freq)
        dft_bins.append(ck)

        current_sampling_freq += increased_sampling_freq

    return dft_bins



def idft_1d(x_axis, dft_bins, sampling_freq):



def show_dft_bins(data):
    plt.bar(range(len(data)), data)
    plt.title('fft freq chart')
    plt.show()


def show_sine_chart(x, curve):
    plt.plot(x, curve)
    plt.show()


if __name__ == "__main__":
    x_axis, my_signals = generate_original_signals(50)
    my_dft = dft_1d(my_signals, len(my_signals), len(my_signals))

    my_bins = []
    for bin in my_dft:
        my_bins.append(bin.amplitude())

    # show_dft_bins(my_bins)

    show_sine_chart(x_axis, my_signals)