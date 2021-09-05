import torch

from Neurons.RNNCell.Model import RNNCellModel

BATCH_SIZE = 1
INPUT_SIZE = 4
HIDDEN_SIZE = 4

# Hello -> ehool

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [0, 1, 3, 3, 2]


one_hot = [[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]]

x_one_hot = [one_hot[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(-1, BATCH_SIZE, INPUT_SIZE)
labels = torch.LongTensor(y_data).view(-1, 1)

model = RNNCellModel(INPUT_SIZE, HIDDEN_SIZE)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(15):
    loss = 0

    optimizer.zero_grad()
    hidden = model.init_hidden()
    print("Predicated String ", end='')

    for input, label in zip(inputs, labels):
        hidden, _ = model(input, hidden)
        loss += criterion(hidden, label)

        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')

    loss.backward()
    optimizer.step()

    print("，Epoch [%d/15] loss=%.4f" % (epoch + 1, loss.item()))

