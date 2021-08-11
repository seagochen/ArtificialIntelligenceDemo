import torch
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    #                     mean       std
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform)

train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist/',
                              train=False,
                              download=True,
                              transform=transform)

test_loader = DataLoader(train_dataset,
                         shuffle=False,
                         batch_size=batch_size)


class FullNeuralNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(784, 512)
        self.layer_2 = torch.nn.Linear(512, 256)
        self.layer_3 = torch.nn.Linear(256, 128)
        self.layer_4 = torch.nn.Linear(128, 64)
        self.layer_5 = torch.nn.Linear(64, 10)

    def forward(self, input_image):
        # each image with 28 x 28 pixels
        x = input_image.view(-1, 784)
        x = functional.relu(self.layer_1(x))
        x = functional.relu(self.layer_2(x))
        x = functional.relu(self.layer_3(x))
        x = functional.relu(self.layer_4(x))
        return self.layer_5(x)


def train(epoch, model, criterion, optimizer):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        # forward, backward, update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch, batch_idx, running_loss / 300))
            running_loss = 0.0


def test(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicated = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicated == labels).sum().item()

    print("Accuracy on test set: %d %%" % (100 * correct / total))


if __name__ == "__main__":

    # full neural network model
    model = FullNeuralNetwork()

    # LOSS function
    criterion = torch.nn.CrossEntropyLoss()

    # parameters optimizer
    # stochastic gradient descent
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

    # training and do gradient descent calculation
    for epoch in range(10):
        # training data
        train(epoch, model, criterion, optimizer)

        # test model
        test(model)
