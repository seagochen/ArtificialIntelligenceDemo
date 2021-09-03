import torch

from Neurons.Utils.Dataset import all_letters


# Find letter index from all_letters, e.g. "a" = 0
def _letter_to_index(letter):
    return all_letters.find(letter)


def line_to_one_hot_tensor(line, max_padding=0):
    """
    Turn a line into a one-hot based tensor (character, one-hot-vector)
    """
    if max_padding >= len(line):
        tensor = torch.zeros(max_padding, len(all_letters))
    else:
        tensor = torch.zeros(len(line), len(all_letters))

    for idx, letter in enumerate(line):
        tensor[idx][_letter_to_index(letter)] = 1
    return tensor


def line_to_ascii_tensor(line, max_padding=0):
    """
    Turn a line into a ascii based tensor (character)
    """
    if max_padding >= len(line):
        tensor = torch.zeros(max_padding)
    else:
        tensor = torch.zeros(len(line))

    for idx, letter in enumerate(line):
        tensor[idx] = ord(letter)
    return tensor


def line_to_index(line: str, data_list: list):
    """
    Turn a line into an index based from dataset
    """
    return data_list.index(line)


def test():
    ascii_based = line_to_ascii_tensor("James", 10)
    one_hot = line_to_one_hot_tensor("James", 7)

    # print(ascii_based)
    # print(ascii_based.shape)
    print(one_hot)
    print(one_hot.shape)

    print(ascii_based)
    print(ascii_based.shape)

    # indx = line_to_index("Japan", ["UK", "USA", "Roma", "Japan"])
    # print(indx)

    # one_hot = one_hot.reshape(7, 1, -1)
    # print(one_hot.shape)


if __name__ == "__main__":
    test()
