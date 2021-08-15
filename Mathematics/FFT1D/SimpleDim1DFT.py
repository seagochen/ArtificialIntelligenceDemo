import numpy as np
from matplotlib import pyplot as plt


def generate_original_signals():
    x_axis = np.linspace(-np.pi, np.pi, 100)
    return x_axis, np.sin(1 * x_axis) + 3 * np.sin(3 * x_axis) + np.sin(7 * x_axis) + np.sin(45 * x_axis)


def correlative_sine_signals_without_phase_and_amplitude(x_axis, signals, basis):
    c_k = 0
    basis_sin = np.sin(x_axis * basis)
    sampling_freq = 50
    steps = len(x_axis) / sampling_freq
    steps = round(steps)

    for i in range(sampling_freq):
        c_k = c_k + signals[i * steps] * basis_sin[i * steps]

    return c_k


def show_dft_bins(data):
    plt.bar(range(len(data)), data)
    plt.title('dft frequencies')
    plt.show()


if __name__ == "__main__":
    x_axis, signals = generate_original_signals()

    # dft_bins = []
    # for i in range(100):
    #     bin = correlative_sine_signals_without_phase_and_amplitude(x_axis, signals, i)
    #     dft_bins.append(bin)
    #     # dft_bins.append(bin)
    #     if bin > 1:
    #         print(i, bin)
    # show_dft_bins(dft_bins)


    basis_sin = np.sin(x_axis)
    sampling_freq = 50
    steps = len(x_axis) / sampling_freq
    steps = round(steps)
    isp = 0

    for i in range(sampling_freq):
        isp = isp + basis_sin[i * steps] * 25

    print(isp / 50)


    # mag_shift = np.sqrt(np.real(fft) ** 2 + np.imag(fft) ** 2)
