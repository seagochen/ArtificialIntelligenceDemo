import torch
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Find letter index from all_letters, e.g. "a" = 0
def _letter_to_index(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <n_letters> Tensor
def _letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][_letter_to_index(letter)] = 1
    return tensor


# Turn a line into a <line_length x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line, padding_len=0):

    if padding_len >= len(line):
        tensor = torch.zeros(padding_len, n_letters)
    else:
        tensor = torch.zeros(len(line), n_letters)

    for li, letter in enumerate(line):
        tensor[li][_letter_to_index(letter)] = 1
    return tensor


def transform(lang: str, name: str, padding=0):
    if padding > 0:
        tensor_lang = line_to_tensor(lang, padding)
        tensor_name = line_to_tensor(name, padding)
    else:
        tensor_lang = line_to_tensor(lang)
        tensor_name = line_to_tensor(name)

    return tensor_name, tensor_lang
