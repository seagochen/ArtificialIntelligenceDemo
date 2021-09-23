from Utilities.DiagnoseTool import load_image_gray
from Utilities.DiagramPlotter import DiagramPlotter

import cv2
import numpy as np


def frequency_noise_analysis(filepath: str):
    img = load_image_gray(filepath)

    # convert byte to float
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    # use NumPy to rapidly shift DFT diagram and prepare for display FFT diagram
    dft_shift = np.fft.fftshift(dft)
    result = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    return [img, result]


def display_result(diagrams):
    plot = DiagramPlotter()

    plot.append_image(diagrams[0], "Original")
    plot.append_image(diagrams[1], "FFT")

    plot.show(1, 2)


if __name__ == "__main__":
    _dft_res = frequency_noise_analysis("../../Data/DIP/DIP3E_CH04_Original_Images/DIP3E_Original_Images_CH04/Fig0421(car_newsprint_sampled_at_75DPI).tif")

    display_result(_dft_res)
