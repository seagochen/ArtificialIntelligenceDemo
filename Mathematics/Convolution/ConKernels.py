import cv2
import numpy as np

from Cv.CvTools.DiagramPlotter import DiagramPlotter


def convolution_kernel(data, kernel):
    # 卷积核计算
    # 将数组从二维转换为一维
    kernel = kernel.flatten()
    data = data.flatten()

    # 将核函数反转后和原始数据进行计算
    kernel = np.flipud(kernel)
    result = kernel * data

    # 返回加和后的值，并四舍五入
    result = round(np.sum(result))
    if result > 255:
        return 255
    elif result < 0:
        return 0
    else:
        return result


def image_convolution(image, kernel):
    # 获取图片的长宽
    width, height = image.shape
    backup = np.zeros(image.shape, np.uint8)

    # 对图片逐像素点遍历
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            # 从图片中取出一个小矩阵，大小跟卷积核大小一致: 3x3
            sub_img = image[i - 1:i + 2, j - 1:j + 2]
            backup[i][j] = convolution_kernel(sub_img, kernel)

    # 返回处理后的图片
    return backup


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


if __name__ == "__main__":
    img = read_image("Data/Illustrations/1.jpeg")
    plt = DiagramPlotter()

    # # 均值
    # kernel = np.array([[1 / 9, 1 / 9, 1 / 9],
    #                    [1 / 9, 1 / 9, 1 / 9],
    #                    [1 / 9, 1 / 9, 1 / 9]])
    # output = image_convolution(img, kernel)
    #
    # # plots images
    # plt.add(img, "original")
    # plt.add(output, "mean")
    # plt.plots(1, 2)
    #
    # # gaussian
    # kernel = np.array([[0.095, 0.118, 0.095],
    #                    [0.118, 0.148, 0.118],
    #                    [0.095, 0.118, 0.095]])
    #
    # output = image_convolution(img, kernel)

    # # plots images
    # plt.clean()
    # plt.add(img, "original")
    # plt.add(output, "gaussian")
    # plt.plots(1, 2)
    #
    # # sharp
    # kernel = np.array([[-1, -1, -1],
    #                    [-1, 9, -1],
    #                    [-1, -1, -1]])
    #
    # output = image_convolution(img, kernel)
    #
    # # plots images
    # plt.clean()
    # plt.add(img, "original")
    # plt.add(output, "enhance")
    # plt.plots(1, 2)

    # gradient
    # kernel = np.array([[0, -1, 0],
    #                    [-1, 0, 1],
    #                    [0, 1, 0]])
    #
    # output = image_convolution(img, kernel)
    #
    # # plots images
    # plt.clean()
    # plt.add(img, "original")
    # plt.add(output, "gradient")
    # plt.plots(1, 2)

    # horizontal
    # kernel = np.array([[-1, -1, -1],
    #                    [0, 0, 0],
    #                    [1, 1, 1]])

    # output = image_convolution(img, kernel)
    #
    # # plots images
    # plt.clean()
    # plt.add(img, "original")
    # plt.add(output, "horizontal")
    # plt.plots(1, 2)
    #
    # # vertical
    # kernel = np.array([[-1, 0, 1],
    #                    [-1, 0, 1],
    #                    [-1, 0, 1]])
    #
    # output = image_convolution(img, kernel)
    #
    # # plots images
    # plt.clean()
    # plt.add(img, "original")
    # plt.add(output, "vertical")
    # plt.plots(1, 2)

    # vertical
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    output = image_convolution(img, kernel)

    # plots images
    plt.clean()
    plt.append_image(img, "original")
    plt.append_image(output, "laplacian")
    plt.show(1, 2)
