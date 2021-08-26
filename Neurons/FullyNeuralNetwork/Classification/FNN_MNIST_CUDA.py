import torch
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


# global definitions
BATCH_SIZE = 64
MNIST_PATH = "../../../Data/MNIST"

# transform sequential
transform = transforms.Compose([
    transforms.ToTensor(),
    #                     mean       std
    transforms.Normalize((0.1307,), (0.3081,))
])

# training dataset
train_dataset = datasets.MNIST(root=MNIST_PATH,
                               train=True,
                               download=True,
                               transform=transform)
# training loader
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=BATCH_SIZE)

# test dataset
test_dataset = datasets.MNIST(root=MNIST_PATH,
                              train=False,
                              download=True,
                              transform=transform)
# test loader
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=BATCH_SIZE)


class FullyNeuralNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # layer definitions
        self.layer_1 = torch.nn.Linear(784, 512)   # 28 x 28 = 784 pixels as input
        self.layer_2 = torch.nn.Linear(512, 256)
        self.layer_3 = torch.nn.Linear(256, 128)
        self.layer_4 = torch.nn.Linear(128, 64)
        self.layer_5 = torch.nn.Linear(64, 10)

    def forward(self, data):
        # transform the image view
        x = data.view(-1, 784)

        # do forward calculation
        x = functional.relu(self.layer_1(x))
        x = functional.relu(self.layer_2(x))
        x = functional.relu(self.layer_3(x))
        x = functional.relu(self.layer_4(x))
        x = self.layer_5(x)

        # return results
        return x


def train(epoch, model, criterion, optimizer):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):

        # convert data to GPU
        inputs, target = data
        inputs = inputs.cuda()
        target = target.cuda()

        # clear gradients
        optimizer.zero_grad()

        # forward, backward, update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # print loss
        running_loss += loss.cpu().item()
        if batch_idx % 300 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch, batch_idx, running_loss / 300))
            running_loss = 0.0


def test(model):
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:
            # convert data to gpu
            images = images.cuda()

            # test
            outputs = model(images)
            _, predicated = torch.max(outputs.data, dim=1)

            # count the accuracy
            total += labels.size(0)
            predicated = predicated.cpu()
            correct += (predicated == labels).sum().item()

    print("Accuracy on test set: %d %%" % (100 * correct / total))


if __name__ == "__main__":

    # full neural network model
    cpu_model = FullyNeuralNetwork()
    gpu_model = cpu_model.cuda()

    # LOSS function
    criterion = torch.nn.CrossEntropyLoss()

    # parameters optimizer
    # stochastic gradient descent
    optimizer = optim.SGD(gpu_model.parameters(), lr=0.1, momentum=0.5)

    # training and do gradient descent calculation
    for epoch in range(5):
        # training data
        train(epoch, gpu_model, criterion, optimizer)

        # test model
        test(gpu_model)
