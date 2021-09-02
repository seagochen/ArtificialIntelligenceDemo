import torch
import torch.optim as optim

from Neurons.RecurrentNeuralNetwork.DataLoader import load_datasets, concatenate_tensors
from Neurons.RecurrentNeuralNetwork.Transform import n_letters
from Neurons.RecurrentNeuralNetwork.RecurrentNeuralNetwork import RNNModel

BATCH_SIZE = 64

# load training and test dataset
train_loader, test_loader = load_datasets("../../Data/NAMES/raw/*.txt")


def train(model, criterion, optimizer):

    # print out predication loss
    running_loss = 0.0

    # train model
    while len(train_loader) > 0:
        # load data from dataset
        x, y = train_loader.load_item(batch_size=BATCH_SIZE, shuffle=True)

        # concatenate list of tensors into one tensor
        x = concatenate_tensors(x)
        y = concatenate_tensors(y)

        # clear the gradients
        optimizer.zero_grad()

        # forward, backward, update
        out, hn = model(x)

        # compress the dimensions of y and output to (-1, 57)
        y = y.view(-1)
        out = out.view(-1)

        loss = criterion(y, out)
        loss.backward()
        optimizer.step()

        #
        # # print loss
        # running_loss += loss.item()
        # if i_batch % 10 == 0:
        #     print('[%5d] loss: %.3f' % (i_batch, running_loss / 10))
        #     running_loss = 0.0



def test(model):
    pass


if __name__ == "__main__":
    # full neural network model
    model = RNNModel(n_letters, n_letters)

    # LOSS function
    criterion = torch.nn.CrossEntropyLoss()

    # parameters optimizer
    # stochastic gradient descent
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # training and optimization
    train(model, criterion, optimizer)

    # test model
    test(model)
