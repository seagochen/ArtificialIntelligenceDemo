#include "OHVCoder.h"
#include "TensorConverter.h"

#include <torch/torch.h>
#include <iostream>
#include <vector>

char all_characters[] = {
        'a', 'b', 'c', 'd', 'e', 'f', 'g',
        'h', 'i', 'j', 'k', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's', 't',
        'u', 'v', 'w', 'x', 'y', 'z',
        'A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        ' ', '.', ',', ';', '\'', '\0'
};

int OHVCoder::letter_to_index(char letter) {
    for (int i = 0; i < strlen(all_characters); i++) {
        if (letter == all_characters[i]) return i;
    }
    return -1;
}

void OHVCoder::line_to_one_hot_tensor
(torch::Tensor& tensor, std::string line, int max_padding)
{
    if (max_padding >= line.size()) {
        int features = static_cast<int>(strlen(all_characters));
        tensor = torch::zeros({ max_padding, features });
    }
    else {
        int linelen = static_cast<int>(line.size());
        int features = static_cast<int>(strlen(all_characters));
        tensor = torch::zeros({ linelen, features });
    }

    for (int idx = 0; idx < line.size(); idx++) {
        char letter = line[idx];
        tensor[idx][letter_to_index(letter)] = 1;
    }
}

void OHVCoder::print_ohv_vector(torch::Tensor& const tensor)
{
    std::cout <<
        tensor.sizes()[0] << " frames " <<
        tensor.sizes()[1] << " features" <<
        std::endl;

    for (int r = 0; r < tensor.sizes()[0]; r++) {
        auto frame = tensor.index({ r,  "..." }).reshape({1, -1});
        auto v = TensorConverter::to<float>(frame);
        std::cout << "frame (" << r << ")    " << v << std::endl;
    }
}
