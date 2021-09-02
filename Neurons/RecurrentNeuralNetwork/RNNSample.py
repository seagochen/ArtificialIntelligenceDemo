import torch


class RNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, sequences=1, batch_size=1, num_layers=1):
        """
        dataset shape (sequences, batch, input_size)

        input of shape (batch, input_size)
        hidden of shape (batch, hidden_size)
        output of shape (batch, hidden_size)
        """
        super(RNN, self).__init__()

        # record some parameters
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequences = sequences

        # our tensor's shape is (N, L, H_in), so we use batch_first
        self.rnn = torch.nn.RNN(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers)

        # input layer shape (batch_size, input_size)
        # output layer shape (batch_size, hidden_size)

    def forward(self, input_data):
        # create hidden layer, dimensions with the same to input_feature
        hidden_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

        # rnn layer do forward calculation
        output_data, hidden_n = self.rnn(input_data, hidden_0)

        # reshape the tensor
        # convert the size from (N, L, H) to (N x L, H)
        # output = output.view(-1, output.size()[2])

        return output_data, hidden_n


def simple_rnn_test():
    batch_size = 1
    sequences = 19
    input_size = 57
    hidden_size = 57

    net = RNN(input_size=input_size, hidden_size=hidden_size,
              batch_size=batch_size, sequences=sequences)

    input = torch.randn(sequences, batch_size, input_size)

    output, hn = net(input)
    print(output.shape)
    print(hn.shape)



def simple_rnn_demo():
    pass


if __name__ == "__main__":
    simple_rnn_test()
