import torch
from torch.utils.data import DataLoader

from Neurons.RecurrentNeuralNetwork.LSTM.LSTMModel import LSTMModel

from Neurons.RecurrentNeuralNetwork.Utils.Dataset import all_letters
from Neurons.RecurrentNeuralNetwork.Utils.Dataset import load_datasets, cherry_pick_items
from Neurons.RecurrentNeuralNetwork.Utils.DataLoader import MyNameDataset
from Neurons.RecurrentNeuralNetwork.Utils.Convert import to_simple_tensor, to_one_hot_based_tensor


# 57 个独热向量
INPUT_SIZE = len(all_letters)

# 输出特征信息64个
HIDDEN_SIZE = 64

# 最终输出18个分类概率
OUTPUT_SIZE = 18

# 单词文字序列长度最大20
SEQUENCE_SIZE = 20

# 一次处理数据10个
BATCH_SIZE = 10


def load_dataset():
    # load data from the txt file
    lang_list, surnames = load_datasets("../../../Data/NAMES/raw/*.txt")

    # split the surname dataset into two parts, the train set and the test set
    train_set, test_set = cherry_pick_items(surnames)

    # wrap the two datasets
    train_set = MyNameDataset(train_set)
    test_set = MyNameDataset(test_set)

    # use torch Dataloader
    train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE)

    # encoding language list
    idx_list = to_simple_tensor(lang_list)

    return lang_list, idx_list, train_loader, test_loader


def convert_data(data):
    # expand parameters
    surnames, regions = data

    # convert surnames to encoded one-hot vectors
    input_x = to_one_hot_based_tensor(surnames)

    # convert regions to integer list
    label_y = to_simple_tensor(regions)

    return input_x, label_y


def train(epoch, model, optimizer, criterion):

    running_loss = 0

    for idx, data in enumerate(train_loader, 0):

        # convert data
        input_x, label_y = convert_data(data)

        # clear the gradients
        optimizer.zero_grad()

        # forward computation
        predicate_y = model(input_x)

        # loss computation
        loss = criterion(predicate_y, label_y)

        # backward propagation
        loss.backward()

        # update network parameters
        optimizer.step()

        # print loss
        running_loss += loss.item()
        if idx % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch, idx, running_loss / 100))
            running_loss = 0


def test(model):
    correct = 0
    total = 0

    with torch.no_grad():

        for idx, data in enumerate(test_loader, 0):
            # convert data
            input_x, label_y = convert_data(data)

            # predicate
            predicate_y = model(input_x)

            # check output
            _, predicated = torch.max(predicate_y.data, dim=1)
            total += label_y.size(0)
            correct += (predicated == label_y).sum().item()

    print("Accuracy on test set: %d %%" % (100 * correct / total))



if __name__ == "__main__":

    # define a model
    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        sequence_size=SEQUENCE_SIZE,
        batch_size=BATCH_SIZE)

    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # majorized function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # load dataset
    languages, language_idx, train_loader, test_loader = load_dataset()

    # Training and testing process
    for epoch in range(10):

        # training
        train(epoch, model, optimizer, criterion)

        # testing
        test(model)