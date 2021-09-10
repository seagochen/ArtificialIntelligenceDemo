from torch.utils.data import Dataset, DataLoader

from Neurons.RecurrentNeuralNetwork.Utils.Dataset import load_datasets, cherry_pick_items


class MyNameDataset(Dataset):

    def __init__(self, dict_data: dict):
        self.x_data = []
        self.y_data = []
        self.languages = []

        for lang, names in dict_data.items():
            for name in names:
                self.x_data.append(name)
                self.y_data.append(lang)

            self.languages.append(lang)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def languages_count(self):
        return len(self.languages)

    def surnames_count(self):
        return len(self.x_data)

    def surname_maxsize(self):
        max_len = 0

        for name in self.x_data:
            if len(name) > max_len:
                max_len = len(name)

        return max_len

    def has_key(self, key):
        return key in self.y_data

    def has_val(self, val):
        return val in self.x_data

    def name_from(self, name):
        index = self.x_data.index(name)
        return self.y_data[index]

    def language_index(self, language):
        return self.languages.index(language)

    def search_language(self, index):
        return self.languages[index]

    def dataset_dict(self):
        dataset = {}

        for language in self.languages:
            dataset[language] = []

        for _, data in enumerate(self):
            name, lang = data
            dataset[lang].append(name)

        return dataset

    def surnames_of_language(self, language):
        return self.dataset_dict()[language]


def unit_test():
    lang_list, surnames = load_datasets("../../Data/NAMES/raw/*.txt")
    train, test = cherry_pick_items(surnames)
    test = MyNameDataset(test)
    print(test.languages_count())
    print(test.surname_maxsize())

    loader = DataLoader(test, shuffle=True, batch_size=10)
    for idx, data in enumerate(loader):
        print(data)


if __name__ == "__main__":
    unit_test()