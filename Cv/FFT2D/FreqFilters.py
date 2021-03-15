def generate_ideal_mask(rows, cols, alpha, beta=0):
    """

    :param rows: rows of spectrum
    :param cols: columns of spectrum
    :param alpha: the frequency upper limit
    :param beta: the frequency lower limit
    :return:
    """

    import numpy as np

    output = np.zeros((rows, cols), np.float32)

    # center coordinate
    cr = rows / 2
    cc = cols / 2

    for r in range(rows):
        for c in range(cols):
            distance = np.sqrt((r - cr) ** 2 + (c - cc) ** 2)
            if alpha > distance >= beta:
                output[r, c] = 1.
            else:
                output[r, c] = 0.

    return output


def generate_butterworth_mask(rows, cols, n, d, flip=False):
    """

    :param rows: rows of spectrum
    :param cols: columns of spectrum
    :param n: the filter adjustment factor
    :param d: the d0
    :return:
    """

    import numpy as np

    output = np.zeros((rows, cols), np.float32)

    # center coordinate
    cr = rows / 2
    cc = cols / 2

    for r in range(rows):
        for c in range(cols):
            distance = np.sqrt((r - cr) ** 2 + (c - cc) ** 2)

            frac = (distance / d) ** (2 * n)
            output[r, c] = 1 / (1 + frac)

    if flip:
        output = 1 - output

    return output


if __name__ == "__main__":
    mask = generate_butterworth_mask(30, 50, 1, 2)

    from Common import *

    plt = PltImageCache()
    plt.add(mask, "butter")
    plt.plots()
