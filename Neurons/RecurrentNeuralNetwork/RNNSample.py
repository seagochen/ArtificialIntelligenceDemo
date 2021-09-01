import torch


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

        # our tensor's shape is (N, L, H_in), so we use batch_first
        self.rnn = torch.nn.RNN(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)

    def forward(self, input_feature):
        # obtain features
        batch_size = input_feature.size()[0]

        # create hidden layer, dimensions with the same to input_feature
        hidden_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        # rnn layer do forward calculation
        output, hidden_n = self.rnn(input_feature, hidden_0)

        # reshape the tensor
        # convert the size from (N, L, H) to (N x L, H)
        # output = output.view(-1, output.size()[2])

        return output


if __name__ == "__main__":

    net = RNN(57, 57)
    x = torch.randn(64, 19, 57)

    output = net(x)
    print(output.size())