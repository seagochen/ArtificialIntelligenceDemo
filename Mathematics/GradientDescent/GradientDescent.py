from CvTools import DataFile as csv
import numpy as np


def gaussian_noise_kernel(x, mu, sigma):
    return np.exp(-1 * ((x - mu) ** 2) / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * sigma)


def load_data():
    x, y = csv.load_pts("Data/csv/gaussian.csv")
    x = np.array(x)
    y = np.array(y)

    return x, y


def cost(y_hats, y_origin):
    return np.sum((y_hats - y_origin) ** 2) / len(y_hats)


def update(mu, sigma, x, y, cost_val, learning_rate):
    # update mu
    y_hat = gaussian_noise_kernel(x, mu - learning_rate, sigma)
    y_hat_cost = cost(y_hat, y)

    if cost_val > y_hat_cost:
        mu = mu - learning_rate
    else:
        mu = mu + learning_rate

    # update sigma
    y_hat = gaussian_noise_kernel(x, mu, sigma - learning_rate)
    y_hat_cost = cost(y_hat, y)

    if cost_val > y_hat_cost:
        sigma = sigma - learning_rate
    else:
        sigma = sigma + learning_rate

    return mu, sigma


if __name__ == "__main__":
    x, y = load_data()
    mu = 100
    sigma = 100
    learning_rate = 1
    iterations = 100

    while iterations > 0:
        predicated_y = gaussian_noise_kernel(x, mu, sigma)
        cost_val = cost(predicated_y, y)

        if cost_val > 0:
            mu, sigma = update(mu, sigma, x, y, cost_val, learning_rate)

        iterations = iterations - 1

    print("mu is", mu, "sigma is", sigma)
