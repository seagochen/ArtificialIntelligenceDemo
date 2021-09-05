import torch

from Neurons.RecurrentNeuralNetwork.RNNCell.RNNCell import RNNCellModel
from Neurons.RecurrentNeuralNetwork.RNNCell.Encoding import line_to_one_hot_tensor, line_to_chaos_tensor
from Neurons.RecurrentNeuralNetwork.RNNCell.Encoding import decode_to_char, n_letters, all_letters

BATCH_SIZE = 1
INPUT_SIZE = n_letters
HIDDEN_SIZE = n_letters

DATA = "hellolele"

inputs = line_to_one_hot_tensor(DATA)
labels = line_to_chaos_tensor(DATA)


def train(epoch, model, optimizer, criterion):
    print("Epoch {} predicated string ".format(epoch), end='')

    loss = 0
    hidden = model.init_hidden()
    optimizer.zero_grad()

    for data, label in zip(inputs, labels):
        hidden = model(data, hidden)
        loss += criterion(hidden, label)

        _, idx = torch.max(hidden, dim=1)
        print(decode_to_char(idx.item()), end='')

    loss.backward()
    optimizer.step()

    print(" loss={:.2f}.".format(loss.item()), end=' ')


def test(model, data):
    print("Test {} to ".format(data), end='')

    # convert the name to one-hot vector
    data = line_to_one_hot_tensor(data)

    # generate predicated values
    hidden = model.init_hidden()
    model.zero_grad()

    # send letter one by one to the model
    for i in range(data.size()[0]):
        x_data = data[i][0].view(1, -1)
        output = model(x_data, hidden)

        _, idx = torch.max(output, dim=1)
        print(decode_to_char(idx.item()), end='')

    print("")


if __name__ == "__main__":

    model = RNNCellModel(INPUT_SIZE, HIDDEN_SIZE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # training
    for epoch in range(20):

        # train the model
        train(epoch, model, optimizer, criterion)

        # test the model
        test(model, DATA)

