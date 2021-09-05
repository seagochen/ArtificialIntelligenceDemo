import torch

# ASCII codes
# all_letters = string.ascii_letters + " .,;'"
all_letters = "aiueo"
n_letters = len(all_letters)


# Find letter index from all_letters, e.g. "a" = 0
def _letter_to_index(letter):
    return all_letters.find(letter)


def line_to_one_hot_tensor(line, max_padding=0):
    """
    Turn a line into a one-hot based tensor (character, one-hot-vector)
    """
    if max_padding >= len(line):
        tensor = torch.zeros(max_padding, 1, n_letters)
    else:
        tensor = torch.zeros(len(line), 1, n_letters)

    for idx, letter in enumerate(line):
        tensor[idx][0][_letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line, max_padding=0):
    """
    Turn a line into a ascii based tensor (character)
    """
    if max_padding >= len(line):
        tensor = torch.zeros(1, max_padding, dtype=torch.long)
    else:
        tensor = torch.zeros(1, len(line), dtype=torch.long)

    for idx, letter in enumerate(line):
        tensor[0][idx] = _letter_to_index(letter)

    return tensor.view(-1, 1)


def line_to_chaos_tensor(line: str, padding=0):
    if padding == 0:
        tensor = torch.zeros(1, len(line), dtype=torch.long)
    else:
        tensor = torch.zeros(1, padding, dtype=torch.long)

    for idx, char in enumerate(line):
        if char == 'a':
            tensor[0][idx] = _letter_to_index('i')
            continue
        if char == 'i':
            tensor[0][idx] = _letter_to_index('u')
            continue
        if char == 'u':
            tensor[0][idx] = _letter_to_index('a')
            continue
        if char == 'e':
            tensor[0][idx] = _letter_to_index('o')
            continue
        if char == 'o':
            tensor[0][idx] = _letter_to_index('e')
            continue

    return tensor.view(-1, 1)


def decode_to_char(data):
    return all_letters[data % len(all_letters)]  # to avoid out of range


def decode_ohv_to_char(tensor):
    _, idx = torch.max(tensor, dim=1)
    return all_letters[idx.item()]


if __name__ == "__main__":
    _data1 = line_to_tensor("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    _data2 = line_to_chaos_tensor("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    print(_data1, _data2.shape)
    print(_data2, _data1.shape)
