import torch
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# Turn a line into a <line_length x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), n_letters)
    for li, letter in enumerate(line):
        tensor[li][letter_to_index(letter)] = 1
    return tensor


if __name__ == "__main__":
    for tensor in line_to_tensor('Jones'):
        print(tensor)
