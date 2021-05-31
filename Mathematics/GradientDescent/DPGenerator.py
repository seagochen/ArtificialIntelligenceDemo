import math
import numpy as np


def gaussian_noise_kernel(x, mu, sigma):
    exp = math.exp(-1 * (math.pow(x - mu, 2) / (2 * math.pow(sigma, 2))))
    peak = (math.sqrt(2 * 3.14159) * sigma)
    return exp / peak


def generate_x_points(pts, start, end):
    return np.linspace(start, end, pts)


if __name__ == "__main__":

    y_pts = gaussian_noise_kernel(generate_x_points(-10, 10, 1000), 1.0, 3)
    print(y_pts)
