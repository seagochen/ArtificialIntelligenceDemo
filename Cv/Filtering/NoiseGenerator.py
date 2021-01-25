import numpy as np
import cv2
import random
import math
from ShowImage import PltImageCache, PltChartCache
import matplotlib.pyplot as plt

def salt_pepper_noise(image, ratio):
    output = np.zeros(image.shape, np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            rand = random.random()
            if rand < ratio:  # salt pepper noise
                if random.random() > 0.5:  # change the pixel to 255
                    output[i][j] = 255
                else:
                    output[i][j] = 0
            else:
                output[i][j] = image[i][j]

    return output


# def gaussian_noise_kernel(x, mu, sigma):
#     exp = math.exp(-1 * (
#                        math.pow(x - mu, 2) / (2 * math.pow(sigma, 2))
#                    ))
#     peak = (math.sqrt(2 * 3.14159) * sigma)
#
#     return exp / peak


def gaussian_noise_kernel(x, mu, sigma):
    return np.exp(-1 * ((x - mu) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)


# if __name__ == "__main__":
#     mu1, sig1 = 0, 1  # standard distribution
#     x1 = np.linspace(mu1 - 5 * sig1, mu1 + 5 * sig1, 50)
#
#     mu2, sig2 = 2, 1  # move the chart to right
#     x2 = np.linspace(mu2 - 5 * sig2, mu2 + 5 * sig2, 50)
#
#     mu3, sig3 = 5, 3.7  # increase the noise coverage
#     x3 = np.linspace(mu3 - 5 * sig3, mu3 + 5 * sig3, 50)
#
#     mu4, sig4 = 5, 0.7  # increase the noise coverage
#     x4 = np.linspace(mu4 - 5 * sig4, mu4 + 5 * sig4, 50)
#
#     # y1, y2, y3, y4 = [], [], [], []
#     # for i in range(50):
#     #     t1 = gaussian_noise_kernel(x1[i], mu1, sig1)
#     #     t2 = gaussian_noise_kernel(x2[i], mu2, sig2)
#     #     t3 = gaussian_noise_kernel(x3[i], mu3, sig3)
#     #     t4 = gaussian_noise_kernel(x4[i], mu4, sig4)
#     #
#     #     y1.append(t1)
#     #     y2.append(t2)
#     #     y3.append(t3)
#     #     y4.append(t4)
#
#     y1 = gaussian_noise_kernel(x1, mu1, sig1)
#     y2 = gaussian_noise_kernel(x2, mu2, sig2)
#     y3 = gaussian_noise_kernel(x3, mu3, sig3)
#     y4 = gaussian_noise_kernel(x4, mu4, sig4)
#
#     plt.plot(x1, y1, 'r', label='mu1, sig1 = 0, 1')
#     plt.plot(x2, y2, 'g', label='mu2, sig2 = 2, 1')
#     plt.plot(x3, y3, 'b', label='mu3, sig3 = 5, 3.7')
#     plt.plot(x4, y4, 'm', label='mu4, sig4 = 5, 0.7')
#     plt.legend()
#     plt.grid()
#     plt.show()


# if __name__ == "__main__":
#     img = cv2.imread("Data/JPG/Asakura Yuu/1.jpg",  cv2.IMREAD_GRAYSCALE)
# 
#     # add salt pepper noise
#     spn_30_img = salt_pepper_noise(img, .30)  # 30%
#     spn_50_img = salt_pepper_noise(img, .50)  # 50%
#     spn_90_img = salt_pepper_noise(img, .90)  # 90%
# 
#     plt = PltImageCache()
#     plt.add(img, "original")
#     plt.add(spn_30_img, "salt pepper 30%")
#     plt.add(spn_50_img, "salt pepper 50%")
#     plt.add(spn_90_img, "salt pepper 90%")
#     plt.plots(2, 2)