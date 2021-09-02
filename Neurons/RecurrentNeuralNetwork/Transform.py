import torch
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Find letter index from all_letters, e.g. "a" = 0
def _letter_to_index(letter):
    return all_letters.find(letter)


# Turn a line into a <char_sequence x batch x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line, padding_len=0):
    if padding_len >= len(line):
        tensor = torch.zeros(padding_len, 1, n_letters)
    else:
        tensor = torch.zeros(len(line), 1, n_letters)

    for li, letter in enumerate(line):
        tensor[li][0][_letter_to_index(letter)] = 1
    return tensor


def transform(lang: str, name: str, padding=0):
    if padding > 0:
        tensor_lang = line_to_tensor(lang, padding)
        tensor_name = line_to_tensor(name, padding)
    else:
        tensor_lang = line_to_tensor(lang)
        tensor_name = line_to_tensor(name)

    return tensor_lang, tensor_name


if __name__ == "__main__":
    _name, _lang = transform("Japanese", "Takahashi")
    print(_name.shape, '\n', _name)
    print(_lang.shape, '\n', _lang)