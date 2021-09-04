import torch


class RNNCellModel(torch.nn.Module):

    """
    input_size – The number of expected features in the input x
    hidden_size – The number of features in the hidden state h
    bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    nonlinearity – The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
    """

    def __init__(self, input_size, hidden_size, batch_size):
        super().__init__()

        self.batch_size = batch_size
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

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


def test():
    from Neurons.Utils.Convert import to_one_hot_based_tensor
    from Neurons.Utils.ModelTest import category_from_output, top_item

    hidden_size = 18  # hidden features, defined by user, not mandatory
    output_size = 18  # in our case, all surnames belong to 18 different regions
    input_size = 57  # one-hot-vector contains 57 separately 0-1 numerics to represent a letter
    sequence_size = 5  # letters in a word with padding length

    # encode a word 'James' to one-hot vector
    # shape of this tensor would be (5, 1, 57)
    words_in_one_hot = to_one_hot_based_tensor(["James"], sequence_size)

    # define a net and do some simple test
    net = RNNCellModel(input_size, output_size, 1)

    # do recurrent computation
    hidden = net.init_hidden()
    for i in range(sequence_size):
        hidden = net(words_in_one_hot[i], hidden)
        print("{:.4f}".format(top_item(hidden)),  category_from_output(hidden))

    print("done")


if __name__ == "__main__":
    test()
