import math
import time

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Neurons.RNNCell.Model import RNNCellModel
from Neurons.Utils.DataLoader import MyNameDataset
from Neurons.Utils.Dataset import load_datasets, cherry_pick_items

# global definitions
INPUT_SIZE = 57  # one-hot-vector contains 57 separately 0-1 numerics to represent a letter
HIDDEN_SIZE = 18  # hidden features, defined by user, not mandatory
BATCH_SIZE = 10  # the count of words in the same time of training and test cycle
SEQUENCE_SIZE = 20  # letters in a word with padding length

# load data and do some preprocessing
lang_list, surnames = load_datasets("../../Data/NAMES/raw/*.txt")
train_loader, test_loader = cherry_pick_items(surnames)
train_loader = DataLoader(MyNameDataset(train_loader), shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(MyNameDataset(test_loader), shuffle=True, batch_size=BATCH_SIZE)


def train(epoch, model, criterion, optimizer):
    pass


def test(model):
    pass


if __name__ == "__main__":

    # define a net and do some simple test
    model = RNNCellModel(INPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE)

    # loss function and optimizer
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # training and do gradient descent calculation
    for epoch in range(5):
        # training data
        train(epoch, model, criterion, optimizer)

        # test model
        test(model)
