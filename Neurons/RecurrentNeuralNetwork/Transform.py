import torch
from Neurons.RecurrentNeuralNetwork.Dataset import all_letters


# Find letter index from all_letters, e.g. "a" = 0
def _letter_to_index(letter):
    return all_letters.find(letter)


def line_to_one_hot_tensor(line, max_padding=0):
    """
    Turn a line into a one-hot based tensor (char_sequence, batch, n_letters)
    """
    if max_padding >= len(line):
        tensor = torch.zeros(max_padding, 1, len(all_letters))
    else:
        tensor = torch.zeros(len(line), 1, len(all_letters))

    for li, letter in enumerate(line):
        tensor[li][0][_letter_to_index(letter)] = 1
    return tensor


def line_to_ascii_tensor(line, max_padding=0):
    """
    Turn a line into a ascii based tensor (char_sequence, batch)
    """
    if max_padding >= len(line):
        tensor = torch.zeros(max_padding, 1)
    else:
        tensor = torch.zeros(len(line), 1)

    for li, letter in enumerate(line):
        tensor[li][0] = ord(letter)
    return tensor


def line_to_index(line: str, data_list: list):
    """
    Turn a line into an index based from dataset
    """
    return data_list.index(line)


def test():
    ascii_based = line_to_ascii_tensor("James", 7)
    one_hot = line_to_one_hot_tensor("James", 7)

    print(ascii_based.shape)
    print(one_hot.shape)

    indx = line_to_index("Japan", ["UK", "USA", "Roma", "Japan"])
    print(indx)


if __name__ == "__main__":
    test()
