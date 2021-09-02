import torch

from Neurons.RecurrentNeuralNetwork.RNNModel import RNNModel


def train_model():
    pass


def test_model():
    pass


def run_model():

    INPUT_SIZE = 57
    HIDDEN_SIZE = 18
    COUNTRIES = 18
    LAYERS = 1
    EPOCH = 10

    net = RNNModel(INPUT_SIZE, HIDDEN_SIZE, COUNTRIES, LAYERS)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(EPOCH):

        train_model()

        test_model()


if __name__ == "__main__":
    run_model()
