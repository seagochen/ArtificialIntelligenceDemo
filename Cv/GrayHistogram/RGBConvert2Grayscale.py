import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(image, title):
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.title(title)
    plt.show()


def show_images(image, title, pos):
    plt.subplot(2, 3, pos)
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.title(title)


def print_matrix(mat):
    # height, width, number of channels in image
    print("matrix size: ", mat.shape)


def rgb_2_gray_normal(rgb_file: str):
    image = cv2.imread(rgb_file)
    row, col, channel = image.shape
    image_gray = np.zeros((row, col))
    for r in range(row):
        for l in range(col):
            # convert rgb image to gray image
            # gray = 0.299⋅R + 0.587⋅G + 0.114⋅B
            image_gray[r, l] = 0.114 * image[r, l, 0] + \
                               0.587 * image[r, l, 1] + \
                               0.299 * image[r, l, 2]

    print_matrix(image_gray)
    return image_gray


def rgb_2_gray_avg(rgb_file: str):
    image = cv2.imread(rgb_file)
    row, col, channel = image.shape
    image_gray = np.zeros((row, col))
    for r in range(row):
        for l in range(col):
            # convert rgb image to gray image
            # gray = 0.333⋅R + 0.333⋅G + 0.333⋅B
            image_gray[r, l] = 0.333 * image[r, l, 0] + \
                               0.333 * image[r, l, 1] + \
                               0.333 * image[r, l, 2]

    print_matrix(image_gray)
    return image_gray


def rgb_2_gray_brightness(rgb_file: str):
    image = cv2.imread(rgb_file)
    row, col, channel = image.shape
    image_gray = np.zeros((row, col))

    for r in range(row):
        for l in range(col):
            image_gray[r, l] = 0.5 * max(image[r, l, 0], image[r, l, 1], image[r, l, 2]) + \
                               0.5 * min(image[r, l, 0], image[r, l, 1], image[r, l, 2])

    print_matrix(image_gray)
    return image_gray


def rgb_2_gray_min(rgb_file: str):
    image = cv2.imread(rgb_file)
    row, col, channel = image.shape
    image_gray = np.zeros((row, col))

    for r in range(row):
        for l in range(col):
            image_gray[r, l] = min(image[r, l, 0], image[r, l, 1], image[r, l, 2])

    print_matrix(image_gray)
    return image_gray


def rgb_2_gray_max(rgb_file: str):
    image = cv2.imread(rgb_file)
    row, col, channel = image.shape
    image_gray = np.zeros((row, col))

    for r in range(row):
        for l in range(col):
            image_gray[r, l] = max(image[r, l, 0], image[r, l, 1], image[r, l, 2])

    print_matrix(image_gray)
    return image_gray


def rgb_2_black(rgb_file: str):
    image = cv2.imread(rgb_file)
    row, col, channel = image.shape
    image_gray = np.zeros((row, col))

    for r in range(row):
        for l in range(col):
            # first of all, convert rgb image to gray
            X = 0.114 * image[r, l, 0] + 0.587 * image[r, l, 1] + 0.299 * image[r, l, 2]

            if X > 90:
                image_gray[r, l] = 255
            else:
                image_gray[r, l] = 0

    print_matrix(image_gray)
    return image_gray


if __name__ == "__main__":
    # show_image(rgb_2_gray_normal("Data/lenna_rgb.tif"), "RGB2Gray General")
    # show_image(rgb_2_gray_avg("Data/lenna_rgb.tif"), "RGB2Gray Average")
    # show_image(rgb_2_gray_brightness("Data/lenna_rgb.tif"), "Brightness First")
    # show_image(rgb_2_gray_min("Data/lenna_rgb.tif"), "Min")
    # show_image(rgb_2_gray_max("Data/lenna_rgb.tif"), "Max")
    # show_image(rgb_2_black("Data/lenna_rgb.tif"), "Black & White")

    norm = rgb_2_gray_normal("Data/lenna_rgb.tif")
    avg = rgb_2_gray_avg("Data/lenna_rgb.tif")
    bright = rgb_2_gray_brightness("Data/lenna_rgb.tif")
    minimum = rgb_2_gray_min("Data/lenna_rgb.tif")
    maximum = rgb_2_gray_max("Data/lenna_rgb.tif")

    show_images(norm, "norm", 1)
    show_images(avg, "avg", 2)
    show_images(bright, "bright", 3)
    show_images(minimum, "min", 4)
    show_images(maximum, "max", 5)

    plt.show()