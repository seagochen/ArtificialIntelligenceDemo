import torch
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


# global definitions
BATCH_SIZE = 64
MNIST_PATH = "../../Data/MNIST"

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


class ConvolutionalNeuralNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()

        """
        Data (64, 1, 28, 28) 
            --Cov1(5, 5)--> (64, 10, 24, 24) 
            --Pool(2, 2)--> (64, 10, 12, 12)
            --Cov1(5, 5)--> (64, 20, 8, 8)
            --Pool(2, 2)--> (64, 20, 4, 4)
            --Flatten/FC--> (64, 320)
            --FC(128, 10)-> (64, 10)
        """

        # layer definitions
        self.conv_1 = torch.nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv_2 = torch.nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.fc_3 = torch.nn.Linear(320, 128)
        self.fc_4 = torch.nn.Linear(128, 10)

        # tools / max pooling
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, data):
        # obtain the batch size
        batch_size = data.size(0)

        # do convolutional computations
        data = functional.relu(self.pool(self.conv_1(data)))
        data = functional.relu(self.pool(self.conv_2(data)))

        # transform the tensor to (batch_size, 320)
        data = data.view(batch_size, -1)

        # FNN
        data = functional.relu(self.fc_3(data))
        data = self.fc_4(data)

        # return results
        return data


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
    cpu_model = ConvolutionalNeuralNetwork()
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
