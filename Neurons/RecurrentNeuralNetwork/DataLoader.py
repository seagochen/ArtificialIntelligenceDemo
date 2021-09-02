import random
import math
import torch
from Neurons.RecurrentNeuralNetwork.Dataset import datasets, cherry_pick_items, max_length
from Neurons.RecurrentNeuralNetwork.Transform import transform


class DataLoader(object):

    def __init__(self, dict_data: dict, max_len=0):

        self.x_data = []
        self.y_data = []

        for lang, names in dict_data.items():
            for surname in names:
                x_ts, y_ts = transform(lang, surname, max_len)
                self.x_data.append(x_ts)
                self.y_data.append(y_ts)

    def __len__(self):
        return len(self.y_data)

    def load_item(self, batch_size=1, shuffle=False):
        output_x = []
        output_y = []

        while self.__len__() > 0 and batch_size > 0:
            if shuffle:
                key = math.ceil(random.random() * self.__len__())
                key = int(key)

                pop_y = self.y_data.pop(key)
                pop_x = self.x_data.pop(key)
            else:
                pop_y = self.y_data.pop()
                pop_x = self.x_data.pop()

            batch_size -= 1

            # append data to list
            output_x.append(pop_x)
            output_y.append(pop_y)

        return output_x, output_y


def load_datasets(path):
    # load original data from files
    _, surnames = datasets(path)
    train_data, test_data = cherry_pick_items(surnames)
    max_len = max_length(surnames)

    train_loader = DataLoader(train_data, max_len)
    test_loader = DataLoader(test_data, max_len)

    return train_loader, test_loader


def concatenate_tensors(tensor_list):
    output = torch.cat(tensor_list, dim=1)
    return output


def test():
    train, test = load_datasets("../../Data/NAMES/raw/*.txt")

    while len(train):
        x, y = train.load_item(batch_size=64)
        print(concatenate_tensors(x).shape == concatenate_tensors(y).shape)


if __name__ == "__main__":
    test()
