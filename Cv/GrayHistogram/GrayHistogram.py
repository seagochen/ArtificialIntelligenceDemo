import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_hist(img):
    # create a matrix to hold histogram information
    histogram = np.zeros(256, dtype=np.uint64)

    # iterate each pixel
    row, col = img.shape
    for r in range(row):
        for c in range(col):
            histogram[img[r, c]] = histogram[img[r, c]] + 1

    # return to caller
    return histogram


def show_charts(hist_data, bar_data):
    # generate bins
    bins = np.arange(256)

    # show bar image on the left
    plt.subplot(1, 2, 1)  # 1 row x 2 cols
    plt.bar(bins, bar_data)
    plt.title('calculated')

    # show hist on the right
    plt.subplot(1, 2, 2)  # 1 row x 2 cols
    plt.hist(hist_data, bins, color='fuchsia', alpha=0.5) #alpha设置透明度，0为完全透明
    plt.title('histogram')

    # plots all
    plt.show()


def show_hist(data):
    # generate bins
    bins = np.arange(256)
    plt.hist(data, bins, color='fuchsia', alpha=0.5) #alpha设置透明度，0为完全透明
    plt.title('histogram')
    plt.show()


def show_images(image, title, pos):
    plt.subplot(1, 2, pos)
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.title(title)


if __name__ == "__main__":
    # load data from file
    img = cv2.imread("Data/lincoln.jpg", cv2.IMREAD_GRAYSCALE)
    # my_hist = calculate_hist(img)

    # show
    # show_charts(img.flatten(), my_hist)

    # flatten = img.flatten()
    low_bound = [0 if elem < 100 else elem for elem in img.flatten()]
    high_bound = [0 if elem >= 160 else elem for elem in img.flatten()]
    low_bound_img = np.array(low_bound, dtype=np.uint8).reshape(img.shape)
    high_bound_img = np.array(high_bound, dtype=np.uint8).reshape(img.shape)

    # plots low and high bound images
    show_images(low_bound_img, "low frequency", 1)
    show_images(high_bound_img, "high frequency", 2)

    plt.show()