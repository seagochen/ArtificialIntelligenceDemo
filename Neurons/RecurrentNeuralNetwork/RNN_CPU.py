import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Neurons.RecurrentNeuralNetwork.DataLoader import load_datasets
from Neurons.RecurrentNeuralNetwork.Transform import n_letters

BATCH_SIZE = 10

# load training and test dataset
train_dataset, test_dataset = load_datasets("../../Data/NAMES/raw/*.txt")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class RNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        (sequential, batch, features)
        """
        super(RNN, self).__init__()

        # record some parameters
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequential = n_letters

        # our tensor's shape is (N, L, H_in), so we use batch_first
        self.rnn = torch.nn.RNN(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)

    def forward(self, input_feature):
        # obtain features

        # create hidden layer, dimensions with the same to input_feature
        hidden_0 = torch.zeros(self.num_layers, self.sequential, self.hidden_size)

        # rnn layer do forward calculation
        output, hidden_n = self.rnn(input_feature, hidden_0)

        # reshape the tensor
        # convert the size from (N, L, H) to (N x L, H)
        # output = output.view(-1, output.size()[2])

        return output


def train(model, criterion, optimizer):

    # print out predication loss
    running_loss = 0.0

    # train model
    for i_batch, batch_data in enumerate(train_loader):
        # derive the data and label from batch_data
        x, y = batch_data

        # clear the gradients
        optimizer.zero_grad()

        # forward, backward, update
        predicated = model(x)
        loss = criterion(y, predicated)
        loss.backward()
        optimizer.step()

        # print loss
        running_loss += loss.item()
        if i_batch % 10 == 0:
            print('[%5d] loss: %.3f' % (i_batch, running_loss / 10))
            running_loss = 0.0



def test(model):
    pass


if __name__ == "__main__":
    # full neural network model
    model = RNN(57, 57)

    # LOSS function
    criterion = torch.nn.CrossEntropyLoss()

    # parameters optimizer
    # stochastic gradient descent
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # training and optimization
    train(model, criterion, optimizer)

    # test model
    test(model)
