import torch
import torch.nn.functional as functional
import torch.optim as optim

from Neurons.RecurrentNeuralNetwork import Dataset as datasets
from Neurons.RecurrentNeuralNetwork import Transform as transform

# DataLoader.load_data("../../Data/NAMES/raw/*.txt")


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(input_size + hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128

# rnn = RNN(Da.n_letters, n_hidden, DataLoader.n_categories)

input_data = transform.letter_to_tensor('A')
hidden = torch.zeros(1, n_hidden)

# output, next_hidden = rnn(input_data, hidden)
# print(output)
