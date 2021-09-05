import torch


class RNNCellModel(torch.nn.Module):

    """
    input_size – The number of expected features in the input x
    hidden_size – The number of features in the hidden state h
    bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    nonlinearity – The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = torch.nn.RNNCell(input_size=self.input_size,
                                     hidden_size=self.hidden_size)

    def forward(self, data, hidden=None):
        """
        Forward computation

        @param data: (batch, input_size), tensor containing input features
        @param hidden: (batch, hidden_size), tensor containing the initial hidden state for each element in the batch.
        Defaults to zero if not provided.
        @return: (batch, hidden_size), tensor containing the next hidden state for each element in the batch

        """
        hidden = self.cell(data, hidden)
        return hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size)
