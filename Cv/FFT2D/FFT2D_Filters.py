import cv2

from Common import *


def load_img_grayscale(file: str):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return image


def shift_spectrum(matrix):
    rows, cols = matrix.shape
    cr = int(rows / 2)
    cc = int(cols / 2)

    quadrant_1 = matrix[cr:rows, 0:cc]
    quadrant_2 = matrix[0:cr, 0:cc]
    quadrant_3 = matrix[0:cr, cc:cols]
    quadrant_4 = matrix[cr:rows, cc:cols]

    temp = quadrant_1.copy()
    quadrant_1 = quadrant_3
    quadrant_3 = temp

    temp = quadrant_2.copy()
    quadrant_2 = quadrant_4
    quadrant_4 = temp

    # yield the updates value back
    output = np.zeros(matrix.shape)
    output[cr:rows, 0:cc] = quadrant_1
    output[0:cr, 0:cc] = quadrant_2
    output[0:cr, cc:cols] = quadrant_3
    output[cr:rows, cc:cols] = quadrant_4

    return np.float32(output)


def do_fft(img):

    # 获取图像的长宽
    rows, cols = img.shape

    # 对长宽进行优化，DFT算法需要输入的数组长度为2^n
    m = cv2.getOptimalDFTSize(rows)
    n = cv2.getOptimalDFTSize(cols)

    # 按照要求，对数据进行填充，不足的部分填充0
    fft2 = np.zeros((m, n, 2), np.float32)
    fft2[:rows, :cols, 0] = img

    # 使用DFT算法，进行快速傅立叶变换
    cv2.dft(fft2, fft2, cv2.DFT_COMPLEX_OUTPUT)

    # 分离复数矩阵
    real_mat, imag_mat = cv2.split(fft2)

    # 返回复数矩阵供下一步操作
    return real_mat, imag_mat


def do_ifft(fft2):

    # 将频率信号转换为原始图像
    ifft2 = np.zeros(fft2.shape[:2], np.float32)
    cv2.dft(fft2, ifft2, cv2.DFT_REAL_OUTPUT | cv2.DFT_INVERSE | cv2.DFT_SCALE)

    # 返回给调用者
    return ifft2


def ideal_filter(real, imag):
    # shift to center
    real_shifted = shift_spectrum(real)
    imag_shifted = shift_spectrum(imag)

    # do frequency filters
    # ...

    # shift back
    real_output = shift_spectrum(real_shifted)
    imag_output = shift_spectrum(imag_shifted)

    # merge back
    complex_mat = cv2.merge((real_output, imag_output))

    # do idft
    return do_ifft(complex_mat)


def butterworth_filter(real, imag):
    # shift to center
    real_shifted = shift_spectrum(real)
    imag_shifted = shift_spectrum(imag)

    # do frequency filters
    # ...

    # shift back
    real_output = shift_spectrum(real_shifted)
    imag_output = shift_spectrum(imag_shifted)

    # merge back
    complex_mat = cv2.merge((real_output, imag_output))

    # do idft
    return do_ifft(complex_mat)


def gaussian_filter(real, imag):
    # shift to center
    real_shifted = shift_spectrum(real)
    imag_shifted = shift_spectrum(imag)

    # do frequency filters
    # ...

    # shift back
    real_output = shift_spectrum(real_shifted)
    imag_output = shift_spectrum(imag_shifted)

    # merge back
    complex_mat = cv2.merge((real_output, imag_output))

    # do idft
    return do_ifft(complex_mat)


def do_filters_demo():

    # load original image
    origin = load_img_grayscale("Data/Illustrations/1.jpg")

    # convert image from spatial to frequencies domain
    real, imag = do_fft(origin)

    # output from ideal filter
    ideal_output = ideal_filter(real, imag)
    butter_output = butterworth_filter(real, imag)
    gaussian_output = gaussian_filter(real, imag)

    # plot images
    plt = PltImageCache()
    plt.add(origin, "origin")
    plt.add(ideal_output, "ideal")
    plt.add(butter_output, "Butterworth")
    plt.add(gaussian_output, "Gaussian")
    plt.plots(2, 2)


if __name__ == "__main__":
    do_filters_demo()
