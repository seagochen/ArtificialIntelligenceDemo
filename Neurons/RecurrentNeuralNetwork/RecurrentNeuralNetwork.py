import torch


class RNNModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size, sequences=1, num_layers=1):
        """
        dataset shape (sequences, batch, input_size)

        input of shape (batch, input_size)
        hidden of shape (batch, hidden_size)
        output of shape (batch, hidden_size)
        """
        super(RNNModel, self).__init__()

        # record some parameters
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequences = sequences

        # our tensor's shape is (N, L, H_in), so we use batch_first
        self.rnn = torch.nn.RNN(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers)

    def forward(self, input_data):
        batch_size = input_data.shape[1]  # (sequence, batch, input)

        # create hidden layer, dimensions with the same to input_feature
        hidden_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        # rnn layer do forward calculation
        output_data, hidden_n = self.rnn(input_data, hidden_0)

        return output_data, hidden_n


def rnn_net_test():
    batch_size = 5
    sequences = 19
    input_size = 57
    hidden_size = 57

    net = RNNModel(input_size=input_size, hidden_size=hidden_size, sequences=sequences)

    input_data = torch.randn(sequences, batch_size, input_size)

    output_data, hn = net(input_data)
    print('shape of input', input_data.shape)
    print('shape of output', output_data.shape)
    print('shape of hidden', hn.shape)


if __name__ == "__main__":
    rnn_net_test()
