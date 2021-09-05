import torch


class RNNLinearImpModel(torch.nn.Module):

    """
    $h_t = tanh(W_{ih}x_t + b_{ih} + W_{hh}h_{(t-1)} + b_{hh})$

    input_size: encoded letter, could be in one-hot-vector, or ascii based code
    hidden_size: hidden state, to recorde previous computed weights
    output_size: output features, for our instance, there are 18 regions to predicate
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLinearImpModel, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(input_size + hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, h0):
        """
        x: encoded letter of word, for instance, 'a' to 'apple'
        ho: previous computed of hidden layer, if no acknowledgement need to
            passed in first recurrent time, transfer a tensor of zero to the net
        """
        combined = torch.cat((x, h0), 1)
        h0 = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, h0

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size)


def category_from_output(output: torch.Tensor):
    """
    convert net output
    """
    from Neurons.Utils.DataLoader import load_datasets
    lang_list, _ = load_datasets("../../Data/NAMES/raw/*.txt")

    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return lang_list[category_i], category_i


def test():
    from Neurons.Utils.Convert import to_one_hot_based_tensor

    hidden_size = 128  # hidden features, defined by user, not mandatory
    output_size = 18  # in our case, all surnames belong to 18 different regions
    input_size = 57  # one-hot-vector contains 57 separately 0-1 numerics to represent a letter
    sequence_size = 5  # letters in a word with padding length

    # encode a word 'James' to one-hot vector
    # shape of this tensor would be (5, 1, 57)
    words_in_one_hot = to_one_hot_based_tensor(["James"], sequence_size)

    # define a net and do some simple test
    net = RNNLinearImpModel(input_size, hidden_size, output_size)

    # do recurrent computation
    hidden = net.init_hidden()
    for i in range(sequence_size):
        out, hidden = net(words_in_one_hot[i], hidden)
        print(category_from_output(out))

    print("done")


if __name__ == "__main__":
    test()
