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


def load_datasets(root: str):
    # if the data not loaded, reload the data
    if len(names_dictionary) == 0 or len(language_list) == 0:
        _load_data(root)

    return language_list, names_dictionary


# max length
def max_length(name_dictionary: dict):
    max_len = 0

    for lang, surnames in name_dictionary.items():
        if max_len < len(lang):
            max_len = len(lang)

        for name in surnames:
            if max_len < len(name):
                max_len = len(name)

    return max_len


def cherry_pick_items(name_dataset=None):
    if name_dataset is None:
        name_dataset = names_dictionary

    train_dataset = {}
    test_dataset = {}

    # iteratively scanning each surnames from different languages
    for lang in language_list:
        surnames = name_dataset[lang]
        train_dataset_temp = []
        test_dataset_temp = []

        # cherry pick item from dataset, separately
        for name in surnames:
            if random.random() < 0.75:
                train_dataset_temp.append(name)
            else:
                test_dataset_temp.append(name)

        # append name list to the different datasets
        train_dataset[lang] = train_dataset_temp
        test_dataset[lang] = test_dataset_temp

    return train_dataset, test_dataset


def test():
    lang_list, surnames = load_datasets("../../Data/NAMES/raw/*.txt")
    print(lang_list)
    print('max len:', max_length(surnames))

    train_sets, test_sets = cherry_pick_items()
    for lan in lang_list:
        train_size = len(train_sets[lan])
        test_size = len(test_sets[lan])
        all_size = len(surnames[lan])

        header = "{} ({})".format(lan, all_size)
        train_per = "{} ({:.2%})".format(train_size, train_size / all_size)
        test_per = "{} ({:.2%})".format(test_size, test_size / all_size)

        result_str = "{0:20} train: {1:15} test: {2:15}".format(header, train_per, test_per)
        print(result_str)


if __name__ == "__main__":
    test()
