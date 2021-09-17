import torch
from torch.nn.utils.rnn import pad_sequence

from Neurons.RecurrentNeuralNetwork.Utils.Transform import line_to_ascii_tensor, line_to_one_hot_tensor, line_to_index


def concatenate_tensors(tensor_list):
    return pad_sequence(tensor_list)


def to_ascii_based_tensors(surnames: list, padding=20):
    tensors = []
    for name in surnames:
        tensor = line_to_ascii_tensor(name, padding)
        tensor = tensor.reshape(padding, 1)
        tensors.append(tensor)

    return concatenate_tensors(tensors)
    # return tensors


def to_one_hot_based_tensor(surnames: list, padding=20):
    tensors = []
    for name in surnames:
        tensor = line_to_one_hot_tensor(name, padding)
        tensors.append(tensor)

    return concatenate_tensors(tensors)
    # return tensors


def to_lang_list_tensor(lang_list: list, languages: list):
    """
    lang_list: 每一个姓名所对应的语言
    languages: 语言所在的列表
    """
    indices = []
    for lang in lang_list:
        index = line_to_index(lang, languages)
        indices.append(index)

    return torch.tensor(indices)


def test():
    tensor1 = to_ascii_based_tensors(['aa', 'bb', 'cc'], 2)
    tensor2 = to_one_hot_based_tensor(['aa', 'bb', 'cc'], 2)

    print(tensor1.shape)
    print(tensor2.shape)

    sequence, batch, letters = tensor1.size()
    for i in range(batch):
        ts = tensor1[:, i, :]
        print(ts)

    sequence, batch, letters = tensor2.size()
    for i in range(batch):
        ts = tensor2[:, i, :]
        print(ts)


if __name__ == "__main__":
    test()