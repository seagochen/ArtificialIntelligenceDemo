import numpy as np

from Utilities import DataFile as csv
from Utilities.DiagramPlotter import DiagramPlotter as Diagram


def gaussian_noise_kernel(x, mu, sigma):
    return np.exp(-1 * ((x - mu) ** 2) / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * sigma)


if __name__ == "__main__":

    x_pts = np.linspace(-10, 20, 1000)
    y_pts = gaussian_noise_kernel(x_pts, 5.0, 3)

    csv.save_pts(x_pts, y_pts, "Data/csv/gaussian.csv")
    xpts, ypts = csv.load_pts("Data/csv/gaussian.csv")

    plot = Diagram()
    plot.append_pts(xpts, ypts)
    plot.show()
