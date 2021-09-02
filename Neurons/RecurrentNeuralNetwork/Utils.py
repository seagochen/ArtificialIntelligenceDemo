import torch
from torch.nn.utils.rnn import pad_sequence

from Neurons.RecurrentNeuralNetwork.Transform import line_to_ascii_tensor, line_to_one_hot_tensor, line_to_index


def concatenate_tensors(tensor_list):
    return pad_sequence(tensor_list)


def to_ascii_based_tensors(surnames, padding=20):
    tensors = []
    for name in surnames:
        tensor = line_to_ascii_tensor(name, padding)
        tensor = tensor.reshape(padding, 1)
        tensors.append(tensor)

    return concatenate_tensors(tensors)
    # return tensors


def to_one_hot_based_tensor(surnames, padding=20):
    tensors = []
    for name in surnames:
        tensor = line_to_one_hot_tensor(name, padding)
        tensors.append(tensor)

    return concatenate_tensors(tensors)
    # return tensors

def to_simple_tensor(languages, dim=0):
    indices = []
    for lang in languages:
        index = line_to_index(lang, languages)
        indices.append(index)

    return torch.tensor(indices)
