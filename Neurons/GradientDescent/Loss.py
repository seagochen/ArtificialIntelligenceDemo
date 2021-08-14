import matplotlib.pyplot as plt
import numpy as np

from Utilities import DataFile as csv


def gaussian_noise_kernel(x, mu, sigma):
    return np.exp(-1 * ((x - mu) ** 2) / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * sigma)


def loss(y_hats, y_origin):
    return np.sum((y_hats - y_origin) ** 2) / len(y_hats)


def loss_mesh(x, y, mu, sigma):
    Z = []

    for mu0 in mu:
        z_mu0 = []
        for sig in sigma:
            y_hat = gaussian_noise_kernel(x, mu0, sig)
            z_mu0.append(loss(y_hat, y))

        Z.append(z_mu0)

    return np.array(Z)


def show_polyhedron():
    x0, y0 = csv.load_pts("Data/csv/gaussian.csv")
    x0 = np.array(x0)
    y0 = np.array(y0)

    mu = np.linspace(1, 10, 1000)
    sigma = np.linspace(1, 10, 1000)
    lost = loss_mesh(x0, y0, mu, sigma)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.contour3D(mu, sigma, lost, 500)
    ax.set_xlabel('mu')
    ax.set_ylabel('sigma')
    ax.set_zlabel('lost')

    plt.show()