import torch
import torch.nn.functional as functional
import torch.optim as optim
from torchnlp import datasets

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"


# global definitions
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of raw per language
category_lines = {}
all_categories = []

for filename in findFiles('data/raw/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines