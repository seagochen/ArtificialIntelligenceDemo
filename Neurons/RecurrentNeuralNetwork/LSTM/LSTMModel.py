import torch


class LSTMModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, sequence_size, batch_size=1, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.output_size = output_size

        self.batch_size = batch_size

        # lstm layer
        self.cell = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers)

        # linear layer for output
        self.linear = torch.nn.Linear(sequence_size * hidden_size, self.output_size)

    def forward(self, inputx):
        """
        forward computation

        @param inputx, tensor of shape (L, N, H_in)
        @param hidden, tensor of shape (D * num_layers, N, H_hidden)
        @param cell,   tensor of shape (D * num_layers, N, H_hidden)

        @return tensor of shape (N, H_out)
        """

        # output tensor (L, N, D * H_out)
        hidden = self.init_hidden()
        cell = self.init_hidden()
        output, _ = self.cell(inputx, (hidden, cell))

        # convert the shape of output to (N, L * H_hidden)
        output = output.reshape(self.batch_size, -1)

        # (N, L * H_hidden) to (N, H_out)
        output = self.linear(output)

        return output

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)


def test():
    input_size = 10
    hidden_size = 20
    output_size = 10

    sequence_size = 5
    batch_size = 10

    # model
    rnn = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        sequence_size=sequence_size,
        batch_size=batch_size)

    # generate input
    tensor_in = torch.randn(sequence_size, batch_size, input_size)
    # tensor_h0 = rnn.init_hidden()
    # tensor_c0 = rnn.init_hidden()

    # output
    output = rnn(tensor_in)
    print(output.shape)


if __name__ == "__main__":
    test()
