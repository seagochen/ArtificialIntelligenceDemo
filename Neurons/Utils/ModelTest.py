import torch
from Neurons.Utils.DataLoader import load_datasets


def category_from_output(output: torch.Tensor):
    """
    convert net output
    """
    lang_list, _ = load_datasets("../../Data/NAMES/raw/*.txt")
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return lang_list[category_i], category_i


def top_item(output: torch.Tensor):
    ts = torch.max(output, dim=1)[0]
    return ts.item()
