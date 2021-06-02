import numpy as np
from CvTools.DiagramPlotter import DiagramPlotter as Diagram


def gaussian_noise_kernel(x, mu, sigma):
    return np.exp(-1 * ((x - mu) ** 2) / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * sigma)


if __name__ == "__main__":

    x_pts = np.linspace(-10, 20, 1000)
    y_pts = gaussian_noise_kernel(x_pts, 5.0, 3)
    
    plot = Diagram()
    plot.append_pts(x_pts, y_pts)
    plot.show()
