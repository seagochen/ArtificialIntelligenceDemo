import torch
from torch.utils.data import Dataset

from Neurons.RecurrentNeuralNetwork.Dataset import datasets, cherry_pick_items, max_length
from Neurons.RecurrentNeuralNetwork.Transform import transform


class NamesDataset(Dataset):

    def __init__(self, dict_data: dict, max_len=0):
        super(NamesDataset, self).__init__()

        self.x_data = []
        self.y_data = []

        for lang, names in dict_data.items():
            for surname in names:
                x_ts, y_ts = transform(lang, surname, max_len)
                self.x_data.append(x_ts)
                self.y_data.append(y_ts)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


def load_datasets(path):
    # load original data from files
    _, surnames = datasets(path)
    train_data, test_data = cherry_pick_items(surnames)
    max_len = max_length(surnames)

    train_data = NamesDataset(train_data, max_len)
    test_data = NamesDataset(test_data, max_len)

    return train_data, test_data


# def convert_dims_to_lnh(dataset):
#     batch_size = dataset.size()[0]
#     sequential = dataset.size()[1]
#     features = dataset.size()[2]
#
#     tensor_data = torch.Tensor(sequential, batch_size, features)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    _train, _test = load_datasets("../../Data/NAMES/raw/*.txt")

    _test_loader = DataLoader(_test, batch_size=64, shuffle=True)

    # test results
    for i_batch, batch_data in enumerate(_test_loader):
        x, y = batch_data

        result_str = "batch {0:2} \ttrain: {1:25} \ttest: {2:25}".format(i_batch, str(x.shape), str(y.shape))
        print(result_str)
