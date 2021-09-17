import torch
from torch.utils.data import DataLoader

from Neurons.RecurrentNeuralNetwork.LSTM.LSTMModel import LSTMModel

from Neurons.RecurrentNeuralNetwork.Utils.Dataset import all_letters
from Neurons.RecurrentNeuralNetwork.Utils.Dataset import load_datasets, cherry_pick_items
from Neurons.RecurrentNeuralNetwork.Utils.DataLoader import MyNameDataset
from Neurons.RecurrentNeuralNetwork.Utils.Convert import to_lang_list_tensor, to_one_hot_based_tensor

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

    return lang_list, train_loader, test_loader


def convert_data(data, languages):
    # expand parameters
    surnames, regions = data

    # convert surnames to encoded one-hot vectors
    input_x = to_one_hot_based_tensor(surnames)

    # convert regions to integer list
    label_y = to_lang_list_tensor(regions, languages)

    # return cuda version to caller
    # return input_x.cuda(), label_y.cuda()
    return input_x, label_y


def train(epoch, model, lang, optimizer, criterion, train_loader):
    running_loss = 0

    for idx, data in enumerate(train_loader, 0):

        # convert data
        input_x, label_y = convert_data(data, lang)

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


def test(model, lang, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, data in enumerate(test_loader, 0):
            # convert data
            input_x, label_y = convert_data(data, lang)

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

    # convert to cuda model
    # model = model.cuda()

    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # majorized function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    # Training and testing process
    for epoch in range(50):

        # load dataset
        languages, train_loader, test_loader = load_dataset()

        # training
        train(epoch, model, languages, optimizer, criterion, train_loader)

        # testing
        test(model, languages, test_loader)

    # save the pytorch model
    torch.save(model, "LSTM_Surname_Classfication_CPU_89acc.ptm")

    # converting to Torch Script via Annotation
    serialized_model = torch.jit.script(model)

    # save the torch script for C++
    serialized_model.save("LSTM_Surname_Classfication_CPU_89acc.pt")