import torch
import string

# ASCII codes
all_letters = string.ascii_letters + " .,;'"


# Find letter index from all_letters, e.g. "a" = 0
def _letter_to_index(letter):
    return all_letters.find(letter)


def line_to_one_hot_tensor(line, max_padding=0):
    """
    Turn a line into a one-hot based tensor (character, one-hot-vector)
    """
    if max_padding >= len(line):
        tensor = torch.zeros(max_padding, 1, len(all_letters))
    else:
        tensor = torch.zeros(len(line), 1, len(all_letters))

    for idx, letter in enumerate(line):
        tensor[idx][0][_letter_to_index(letter)] = 1
    return tensor


def line_to_ascii_tensor(line, max_padding=0):
    """
    Turn a line into a ascii based tensor (character)
    """
    if max_padding >= len(line):
        tensor = torch.zeros(1, max_padding)
    else:
        tensor = torch.zeros(1, len(line))

    for idx, letter in enumerate(line):
        tensor[0][idx] = ord(letter) - 65

    return tensor


def line_to_chaos_ascii_tensor(line: str, padding=0):
    if padding == 0:
        tensor = torch.zeros(1, len(line))
    else:
        tensor = torch.zeros(1, padding)

    for idx, char in enumerate(line):
        if char == 'a':
            tensor[0][idx] = ord('i') - 65
            continue
        if char == 'i':
            tensor[0][idx] = ord('u') - 65
            continue
        if char == 'u':
            tensor[0][idx] = ord('a') - 65
            continue
        if char == 'e':
            tensor[0][idx] = ord('o') - 65
            continue
        if char == 'o':
            tensor[0][idx] = ord('e') - 65
            continue

        if char == 'A':
            tensor[0][idx] = ord('I') - 65
            continue
        if char == 'I':
            tensor[0][idx] = ord('U') - 65
            continue
        if char == 'U':
            tensor[0][idx] = ord('A') - 65
            continue
        if char == 'E':
            tensor[0][idx] = ord('O') - 65
            continue
        if char == 'O':
            tensor[0][idx] = ord('E') - 65
            continue

        tensor[0][idx] = ord(char) - 65

    return tensor


if __name__ == "__main__":
    _data1 = line_to_ascii_tensor("ABCDEFG")
    _data2 = line_to_chaos_ascii_tensor("ABCDEFG")

    print(_data1, _data2.shape)
    print(_data2, _data1.shape)
