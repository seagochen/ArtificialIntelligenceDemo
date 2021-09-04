import random
import torch
from Neurons.Utils.DataLoader import load_datasets
from Neurons.Utils.Dataset import all_letters

all_categories, category_lines = load_datasets("../../Data/NAMES/raw/*.txt")


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, len(all_letters))
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, len(all_letters))
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_example():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def test():
    for i in range(10):
        category, line, category_tensor, line_tensor = random_training_example()
        print('category =', category, '/ line =', line)


if __name__ == "__main__":
    test()
