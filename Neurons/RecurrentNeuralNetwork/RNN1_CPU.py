import torch
import torch.nn.functional as functional
import torch.optim as optim

BATCH_SIZE = 10


class RNN(torch.nn.Module):

    def __init__(self, input_features, hidden_features, output_features):

        """
        (sequential, batch, features)
        """
        super(RNN, self).__init__()

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features

        self.batch_size = BATCH_SIZE

        self.rnn = torch.nn.RNN(input_size=input_features,
                                hidden_features=hidden_features)

    def forward(self, input_feature):
        hidden_layer = torch.zeros(self.batch_size, self.hidden_features)
        output, hidden = self.rnn(input_feature, hidden_layer)

        # reshape the tensor
        output = output.view(-1, self.hidden_features)  # --> (sequential x batch_size, hidden_features)

        return output



if __name__ == "__main__":

    # full neural network model
    model = RNN()

    # LOSS function
    criterion = torch.nn.CrossEntropyLoss()

    # parameters optimizer
    # stochastic gradient descent
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

    # training and do gradient descent calculation
    for epoch in range(5):
        # training data
        train(epoch, model, criterion, optimizer)

        # test model
        test(model)
