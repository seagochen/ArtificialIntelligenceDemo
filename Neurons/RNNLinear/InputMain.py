import math
import time

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Neurons.RNNLinear.Model import RNNLinearImpModel, category_from_output
from Neurons.RNNLinear.RandomChoiceTest import random_training_example
from Neurons.Utils.Convert import to_one_hot_based_tensor
from Neurons.Utils.DataLoader import MyNameDataset
from Neurons.Utils.Dataset import load_datasets, cherry_pick_items

from Utilities.DiagramPlotter import DiagramPlotter

# global definitions
INPUT_SIZE = 57  # one-hot-vector contains 57 separately 0-1 numerics to represent a letter
HIDDEN_SIZE = 128  # hidden features, defined by user, not mandatory
OUTPUT_SIZE = 18  # in our case, all surnames belong to 18 different regions
BATCH_SIZE = 10  # the count of words in the same time of training and test cycle
SEQUENCE_SIZE = 20  # letters in a word with padding length

# load data and do some preprocessing
lang_list, surnames = load_datasets("../../Data/NAMES/raw/*.txt")
train_loader, test_loader = cherry_pick_items(surnames)
train_loader = DataLoader(MyNameDataset(train_loader), shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(MyNameDataset(test_loader), shuffle=True, batch_size=BATCH_SIZE)


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# def train(batch_label, batch_input, rnn, criterion):
#     """
#     batch_label, shape is (batch)
#     batch_input, shape is (sequence, batch, one-hot)
#     """
#
#     # If you set this too high, it might explode. If too low, it might not learn
#     learning_rate = 0.005
#
#     # Create all zero tensor for first cycle
#     hidden = rnn.init_hidden()
#     output = None
#
#     # get features from batch_input
#     sequence, batch, vectors = batch_input.size()
#
#     # Training cycle:
#     # Each letter one by one sent to the net will produce an output
#     # and updated hidden state.
#     # However, we will ignore the middle status, just consider about
#     # the final output as the result and do backward and loss computation
#     for i in range(sequence):
#         output, hidden = rnn(batch_input[i], hidden)
#
#     # Compute loss
#     loss = criterion(output, batch_label)
#     loss.backward()
#
#     # Update net's parameters
#     # optimizer.step()
#     # Add parameters' gradients to their values, multiplied by learning rate
#     for p in rnn.parameters():
#         p.data.add_(p.grad.data, alpha=-learning_rate)
#
#     # return result and current cycle's loss value
#     return output, loss.item()

def train(category_tensor, line_tensor, rnn, criterion):
    hidden = rnn.init_hidden()

    rnn.zero_grad()
    output = None

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    learning_rate = 0.005

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def rnn_demo():
    # define a net and do some simple test
    model = RNNLinearImpModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    # loss function and optimizer
    criterion = torch.nn.NLLLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_iters = 100000
    print_every = 5000
    plot_every = 1000
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = random_training_example()
        output, loss = train(category_tensor, line_tensor, model, criterion)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
                iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    plt.figure()
    plt.plot(all_losses)
    plt.show()

    return model


def rnn_predict(model, surname):
    # convert the name to one-hot vector
    inputs = to_one_hot_based_tensor([surname], SEQUENCE_SIZE)

    # generate predicated values
    batch_size = inputs.size()[1]
    hidden = model.init_hidden(batch_size)

    # send letter one by one to the model
    output = None
    for i in range(inputs.size()[0]):
        # forward computation
        output, hidden = model(inputs[i], hidden)

    return output


def input_test(model):

    while True:
        # get user input
        surname = input("Enter a surname:")

        # if surname is e, stop the program
        if surname == "e":
            break

        # convert the surname to one-hot vector
        result = rnn_predict(model, surname)

        # show predicated value
        lang, _ = category_from_output(result)
        print("Is {} a {} name?".format(surname, lang))


if __name__ == "__main__":
    input_test(rnn_demo())
