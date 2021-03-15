import cv2

from Common import *


def load_img_grayscale(file: str):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return image


def shift_spectrum(gamma_matrix):
    rows, cols = gamma_matrix.shape
    cr = int(rows / 2)
    cc = int(cols / 2)

    quadrant_1 = gamma_matrix[cr:rows, 0:cc]
    quadrant_2 = gamma_matrix[0:cr, 0:cc]
    quadrant_3 = gamma_matrix[0:cr, cc:cols]
    quadrant_4 = gamma_matrix[cr:rows, cc:cols]

    temp = quadrant_1.copy()
    quadrant_1 = quadrant_3
    quadrant_3 = temp

    temp = quadrant_2.copy()
    quadrant_2 = quadrant_4
    quadrant_4 = temp

    # yield the updates value back
    output = np.zeros(gamma_matrix.shape)
    output[cr:rows, 0:cc] = quadrant_1
    output[0:cr, 0:cc] = quadrant_2
    output[0:cr, cc:cols] = quadrant_3
    output[cr:rows, cc:cols] = quadrant_4

    return output


def gen_spectrum(real_matrix, imag_matrix):
    # create a gamma matrix
    gamma_matrix = np.zeros(real_matrix.shape, dtype=np.float32)
    cv2.magnitude(real_matrix, imag_matrix, gamma_matrix)
    cv2.log(gamma_matrix + 1, gamma_matrix)

    # normalize
    output = np.zeros(real_matrix.shape, dtype=np.float32)
    cv2.normalize(gamma_matrix, output, 0, 1, cv2.NORM_MINMAX)

    return output


def do_fft():
    # 获取原始图像的灰度图
    img = load_img_grayscale("Data/Illustrations/1.jpg")

    # 获取图像的长宽
    # C++代码里，需要用 img.rows, img.cols来获取对应的数据
    rows, cols = img.shape

    # 对长宽进行优化，DFT算法需要输入的数组长度为2^n
    m = cv2.getOptimalDFTSize(rows)
    n = cv2.getOptimalDFTSize(cols)

    # 按照要求，对数据进行填充，不足的部分填充0
    # C++中使用
    # cv2.copyMakeBorder(src, dst, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT)
    fft2 = np.zeros((m, n, 2), np.float32)
    fft2[:rows, :cols, 0] = img

    # 为DFT创建实数部和虚数部
    # C++对应的方法：
    # Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    # Mat complex;
    # cv2::merge(planes, 2, complex):
    # dft(complexImg, complexImg, cv2.DFT_COMPLEX_OUTPUT)

    # 使用DFT算法，进行快速傅立叶变换
    cv2.dft(fft2, fft2, cv2.DFT_COMPLEX_OUTPUT)

    # 分离复数矩阵
    real_mat, imag_mat = cv2.split(fft2)

    # 生成频率数据，由于一部分数据属于不可见数据，所以需要进行gamma变换
    # 并执行归一化操作
    gamma_spectrum = gen_spectrum(real_mat, imag_mat)

    # 为gamma图执行中央转换
    # 方法是1，3象限对调，2，4象限对调
    shift_spectrum = shit_spectrum(gamma_spectrum)

    # 把原始图像、FFT频率图、转化后的频率图全部显示出来
    plt = PltImageCache()
    plt.add(img, "origin")
    plt.add(gamma_spectrum, "dft computed")
    plt.add(shift_spectrum, "dft shifted")
    plt.plots(1, 3)

    # 返回复数矩阵供下一步操作
    return fft2


def do_ifft(fft2):

    # 获取原始图像的灰度图
    img = load_img_grayscale("Data/Illustrations/1.jpg")

    ifft2 = np.zeros(fft2.shape[:2], np.float32)
    cv2.dft(fft2, ifft2, cv2.DFT_REAL_OUTPUT | cv2.DFT_INVERSE | cv2.DFT_SCALE)

    # 把原始图像、FFT频率图、转化后的频率图全部显示出来
    plt = PltImageCache()
    plt.add(img, "original")
    plt.add(ifft2, "idft")
    plt.plots(1, 2)


if __name__ == "__main__":
    dft_matrix = do_fft()
    do_ifft(dft_matrix)