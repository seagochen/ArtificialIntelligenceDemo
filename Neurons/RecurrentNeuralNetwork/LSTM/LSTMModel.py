import torch
import torch.nn.functional as functional


class LSTMModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, batch_size, sequence_size, num_layers=1):
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

    def forward(self, input_x):
        """
        forward computation

        @param input_x, tensor of shape (L, N, H_in)
        @return tensor of shape (N, H_out)
        """

        # get dimension from input_x
        _, batch, features = input_x.size()

        # hidden, tensor of shape (D * num_layers, N, H_hidden)
        # hidden = self.init_zeros(batch)
        hidden = torch.zeros(self.num_layers, batch, self.hidden_size)

        # cell, tensor of shape (D * num_layers, N, H_hidden)
        # cell = self.init_zeros(batch)
        cell = torch.zeros(self.num_layers, batch, self.hidden_size)

        # output tensor (L, N, D * H_hidden)
        output, _ = self.cell(input_x, (hidden, cell))

        # convert the shape of output to (N, L * H_hidden)
        # hidden = self.convert_hidden_shape(output, torch.tensor(batch))
        tensor_list = []
        for i in range(batch):
            ts = output[:, i, :].reshape(1, -1)
            tensor_list.append(ts)

        final_output = torch.cat(tensor_list)

        # (N, L * H_hidden) to (N, H_out)
        output = self.linear(final_output)
        output = functional.relu(output)

        return output


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
    print("(L, N, Input_Features) ---> (N, Possibilities)", output.shape)


if __name__ == "__main__":
    test()
