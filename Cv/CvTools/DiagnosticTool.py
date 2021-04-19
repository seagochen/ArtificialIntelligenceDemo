import numpy as np
import cv2


def hist(img):
    histogram = np.zeros(256, dtype=np.uint64)

    row, col = img.shape
    for r in range(row):
        for c in range(col):
            histogram[img[r, c]] = histogram[img[r, c]] + 1

    return histogram


def load_image(file: str):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return image
