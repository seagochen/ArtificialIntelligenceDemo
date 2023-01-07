import numpy as np
from matplotlib import pyplot as plt

def wavelet_ricker(frequency, num_samples):
    t = np.linspace(-1, 1, num_samples)
    wavelet = (1 - 2 * np.pi**2 * frequency**2 * t**2) * np.exp(-np.pi**2 * frequency**2 * t**2)
    return wavelet


def main():
    # Generate a ricker wavelet with 1024 samples and a frequency of 10hz
    wavelet = wavelet_ricker(10, 1024)

    # Plot the wavelet
    plt.plot(wavelet)
    plt.title('Ricker wavelet')
    plt.show()

if __name__ == '__main__':
    main()