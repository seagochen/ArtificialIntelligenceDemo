import torch
from torch.utils.data import DataLoader

from Neurons.RecurrentNeuralNetwork.Utils.Convert import to_lang_list_tensor, to_one_hot_based_tensor
from Neurons.RecurrentNeuralNetwork.Utils.DataLoader import MyNameDataset
from Neurons.RecurrentNeuralNetwork.Utils.Dataset import load_datasets, cherry_pick_items

# 一次处理数据10个
BATCH_SIZE = 10


def load_dataset():
    # load data from the txt file
    lang_list, surnames = load_datasets("../../../Data/NAMES/raw/*.txt")

    # split the surname dataset into two parts, the train set and the test set
    train_set, test_set = cherry_pick_items(surnames)

    # wrap the two datasets
    train_set = MyNameDataset(train_set)
    test_set = MyNameDataset(test_set)

    # use torch Dataloader
    train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE)

    return lang_list, train_loader, test_loader


def convert_data(name, languages):
    # expand parameters
    # surnames, regions = data

    # convert surnames to encoded one-hot vectors
    input_x = to_one_hot_based_tensor([name])

    # convert regions to integer list
    # label_y = to_lang_list_tensor([region], languages)

    # return cuda version to caller
    return input_x


def lang_with_perhaps(predict, lang):
    for i in range(len(lang)):
        print("{0:20}: {1:.2}".format(lang[i], predict[:, i].item()))


def test(model, lang, name):
    with torch.no_grad():
        # convert data
        input_x = convert_data(name, lang)

        # predicate
        predicate_y = model(input_x)

        # check output
        _, predicated = torch.max(predicate_y.data, dim=1)

        # print out predication
        print("My predicate is ", lang[predicated.item()])

        # print out perhaps
        lang_with_perhaps(predicate_y, lang)


if __name__ == "__main__":

    # define a model
    model = torch.load("LSTM_Surname_Classfication_CPU_89acc.ptm")

    # Training and testing process
    # for epoch in range(10):

    # load dataset
    languages, train_loader, test_loader = load_dataset()

    while True:
        name = input("Name? ")
        if name == "exit":
            break

        # testing
        test(model, languages, name)

    print("Adios~")
