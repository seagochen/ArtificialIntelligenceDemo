def haar_dwt2(image, level):
    """
    Compute the Haar wavelet decomposition of an image.
    """
    # Check if the image is a power of 2
    if np.log2(image.shape[0]) % 1 > 0 or np.log2(image.shape[1]) % 1 > 0:
        raise ValueError("The image size must be a power of 2.")
    # Check if the level is valid
    if level < 0 or level > np.log2(image.shape[0]):
        raise ValueError("The level must be between 0 and log2(image.shape[0]).")
    # Compute the wavelet decomposition
    image = image.copy()
    wavelet = []
    for i in range(level):
        # Compute the wavelet coefficients
        wavelet.append(image[1::2, 1::2] - image[::2, 1::2] - image[1::2, ::2] + image[::2, ::2])
        # Compute the scaling coefficients
        image = (image[1::2, 1::2] + image[::2, 1::2] + image[1::2, ::2] + image[::2, ::2]) / 4
    # Return the wavelet decomposition
    return image, wavelet

def haar_idwt2(scaling, wavelet):
    """
    Compute the inverse Haar wavelet decomposition of an image.
    """
    # Check if the level is valid
    if len(wavelet) != scaling.shape[0] - 1:
        raise ValueError("The level must be len(wavelet) = scaling.shape[0] - 1.")
    # Compute the inverse wavelet decomposition
    image = scaling.copy()
    for i in range(len(wavelet)):
        # Compute the scaling coefficients
        image = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
        # Compute the wavelet coefficients
        image[1::2, 1::2] += wavelet[-i-1]
        image[::2, 1::2] -= wavelet[-i-1]
        image[1::2, ::2] -= wavelet[-i-1]
        image[::2, ::2] += wavelet[-i-1]
    # Return the inverse wavelet decomposition
    return image