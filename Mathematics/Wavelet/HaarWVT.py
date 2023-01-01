import math

def discrete_wavelet_transform(data, depth):
    # Haar Discrete Wavelet Transform, depth passes
    depth = min(int(math.log2(len(data))), depth)  # Limit depth to number of passes needed to transform entire data list
    details = []  # Initialize empty list to store details at different scales

    # Loop through number of passes
    for d in range(depth):
        # Initialize empty lists for approximations and details for current pass
        approximations, details_for_pass = [], []
        # Loop through data by pairs of values
        for i in range(0, len(data), 2):
            # Append average of pair to list of approximations
            approximations.append(sum(data[i:i+2]) / 2)
            # Append difference between first value of pair and average to list of details
            details_for_pass.append(data[i] - approximations[-1])
        # Add current details to beginning of list
        details = details_for_pass + details
        # Replace data with approximations for next pass
        data = approximations

    return approximations, details  # Return approximations and details at different scales


def inverse_discrete_wavelet_transform(approximations, details):
    # Haar Inverse Discrete Wavelet Transform, depth passes
    N = len(approximations + details)  # Calculate length of original data
    sums = [sum([[k] * (N // len(approximations)) for k in approximations], [])]  # Initialize list of sums with approximations
    bs, i = N // 2, 0  # Initialize block size and index

    # Loop until block size becomes 0 or all details have been processed
    while bs > 0 and i < len(details):
        # Initialize list of values for current block of details
        values = []
        # Loop through current block of details
        for a in details[::-1][i:i+bs]:
            # Append alternating negative and positive values to list of values
            values.extend([-a] * (N // (2 * bs)))
            values.extend([a] * (N // (2 * bs)))
        # Reverse list of values and add it to list of sums
        sums.append(values[::-1])
        # Move to next block of details
        i += bs
        # Divide block size by 2
        bs //= 2

    # Calculate reconstructed data by summing values at each index in list of sums
    rec_data = [sum([s[j] for s in sums]) for j in range(N)]
    return rec_data  # Return reconstructed data


def main():
    # Generate a sine wave with 1024 samples
    data = [math.sin(2 * math.pi * 10 * i / 1024) for i in range(1024)]

    # Perform Haar Discrete Wavelet Transform
    approximations, details = discrete_wavelet_transform(data, 10)

    # Perform Haar Inverse Discrete Wavelet Transform
    rec_data = inverse_discrete_wavelet_transform(approximations, details)

    # Print every pair of values in original data and reconstructed data
    for i in range(len(data)):
        # Keep 4 decimal places
        original, reconstructed = f'{data[i]:.4f}', f'{rec_data[i]:.4f}'
        print(i, original, reconstructed)

if __name__ == '__main__':
    main()