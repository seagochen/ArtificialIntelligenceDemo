import torch

from torch.utils.data import DataLoader

from Neurons.Utils.Dataset import load_datasets, cherry_pick_items
from Neurons.Utils.DataLoader import MyNameDataset
from Neurons.Utils.Convert import to_ascii_based_tensors
from Neurons.Utils.Convert import to_simple_tensor
from Neurons.RNNLow.Model import RNN

# global definitions
INPUT_SIZE = 1
HIDDEN_SIZE = 1
BATCH_SIZE = 10
VECTOR_SIZE = 20

# load data and do some preprocessing
lang_list, surnames = load_datasets("../../Data/NAMES/raw/*.txt")
train_loader, test_loader = cherry_pick_items(surnames)
train_loader = DataLoader(MyNameDataset(train_loader), shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(MyNameDataset(test_loader), shuffle=True, batch_size=BATCH_SIZE)


def train_model(train_loader, model):

    # loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    running_loss = 0.0

    # training cycle
    for idx, data in enumerate(train_loader):
        inputs, labels = data

        # convert inputs and labels
        # shape of vector (VECTOR_SIZE, BATCH_SIZE, INPUT_SIZE)
        # shape of labels (BATCH_SIZE)
        inputs = to_ascii_based_tensors(inputs, padding=VECTOR_SIZE)
        labels = to_simple_tensor(labels)

        # clear the gradients
        optimizer.zero_grad()

        # forward, backward, update
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        # print out debug message
        running_loss += loss.item()
        if idx % 64 == 0:
            print('[%5d] loss: %.3f' % (idx, running_loss / 64))
            running_loss = 0.0


def test_model(test_loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:

            # convert key-value to tensor
            inputs = to_ascii_based_tensors(inputs, padding=VECTOR_SIZE)
            labels = to_simple_tensor(labels)

            out = model(inputs)
            _, predicated = torch.max(out.data, dim=1)
            total += labels.size(0)
            correct += (predicated == labels).sum().item()

    print("Accuracy on test set: %d %%" % (100 * correct / total))


if __name__ == "__main__":

    # declare a rnn model
    model = RNN(INPUT_SIZE, HIDDEN_SIZE)

    # run training cycle
    for i in range(1, 10 + 1):
        train_model(train_loader, model)
        test_model(test_loader, model)
