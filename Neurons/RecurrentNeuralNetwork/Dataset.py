import glob
import os
import random
import string
from io import open

import unicodedata

# ASCII codes
all_letters = string.ascii_letters + " .,;'"

# Build the category_lines dictionary, a list of raw per language
names_dictionary = {}

# Language species list
language_list = []

# train dataset
train_dataset = {}

# test dataset
test_dataset = {}


def find_files(path): return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def _unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def _read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [_unicode_to_ascii(line) for line in lines]


def _load_data(path):
    for filename in find_files(path):
        category = os.path.splitext(os.path.basename(filename))[0]
        language_list.append(category)
        lines = _read_lines(filename)
        names_dictionary[category] = lines


def _cherry_pick_items():
    # iteratively scanning each surnames from different languages
    for lang in language_list:
        surnames = names_dictionary[lang]
        train_dataset_temp = []
        test_dataset_temp = []

        # cherry pick item from dataset, separately
        for name in surnames:
            if random.random() < 0.8:
                train_dataset_temp.append(name)
            else:
                test_dataset_temp.append(name)

        # append name list to the different datasets
        train_dataset[lang] = train_dataset_temp
        test_dataset[lang] = test_dataset_temp


def names(root: str, train: bool):
    # if the data not loaded, reload the data
    if len(names_dictionary) == 0 or len(language_list) == 0:
        _load_data(root)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        _cherry_pick_items()

    if train:
        return train_dataset
    else:
        return test_dataset


if __name__ == "__main__":
    test = names("../../Data/NAMES/raw/*.txt", False)
    train = names("../../Data/NAMES/raw/*.txt", True)

    print(len(test['Chinese']))
    print(len(train['Chinese']))