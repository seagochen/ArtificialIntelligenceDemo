import cv2
import numpy as np

from Utilities.DiagnoseTool import load_image_gray
from Utilities.DiagramPlotter import DiagramPlotter


def degradation_kernel(dft, a, b, t):
    # derive width, height, channel
    width, height, _ = dft.shape

    # center pointer
    p = width / 2 + 1.0
    q = height / 2 + 1.0

    # generate an empty kernel
    kernel = np.zeros((width, height), dtype=np.float32)

    # generate turbulence kernel
    for u in range(width):
        for v in range(height):
            t_pi = t / np.pi * (u * a + v * b)


            power = -k * np.power((u - p) ** 2 + (v - q) ** 2, 5 / 6)
            kernel[u, v] = np.power(np.e, power)

    return kernel


def update_dft_with_degradation(dft, kernel):

    # derive width, height, channel
    width, height, _ = dft.shape

    # shift dft
    dft_backup = np.fft.fftshift(dft)

    # apply the kernel
    dft_backup[:, :, 0] = dft_backup[:, :, 0] * kernel
    dft_backup[:, :, 1] = dft_backup[:, :, 1] * kernel

    # shift back
    dft_backup = np.fft.fftshift(dft_backup)

    return dft_backup


def motion_deterioration(img, a, b, t):

    # convert byte to float
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    # generate kinetic degradation
    kernel = degradation_kernel(dft, a, b, t)

    # apply kernel
    final_dft = update_dft_with_degradation(dft, kernel)

    # convert dft image back
    final_img = cv2.idft(final_dft, flags=cv2.DFT_COMPLEX_INPUT | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # return
    return final_img


def display_result(diagrams):
    plot = DiagramPlotter()

    plot.append_image(diagrams[0], "Original")
    plot.append_image(diagrams[1], "a,b=0.1, T=1")

    plot.show(1, 2)


if __name__ == "__main__":

    # load image from file
    img = load_image_gray(
        "../../Data/DIP/DIP3E_CH05_Original_Images/DIP3E_CH05_Original_Images/Fig0526(a)(original_DIP).tif")
    img_with_ab0d1_t1 = motion_deterioration(img, 0.1, 0.1, 1)
    display_result((img, img_with_ab0d1_t1))