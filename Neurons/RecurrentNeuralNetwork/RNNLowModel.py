import torch


class RNNModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
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

        # our tensor's shape is (N, L, H_in), so we use batch_first
        self.rnn = torch.nn.RNN(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers)

    def forward(self, input_data):
        batch_size = input_data.shape[1]  # (sequence, batch, input)

        # create hidden layer, dimensions with the same to input_feature
        hidden_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        # rnn layer do forward calculation
        out, hidden_n = self.rnn(input_data, hidden_0)

        # reshape output to (N, C)
        return out.reshape(batch_size, -1)
        # output_data = self.softmax(output_data)
