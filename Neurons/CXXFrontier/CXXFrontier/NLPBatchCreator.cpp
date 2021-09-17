#include "NLPBatchCreator.h"
#include "OHVCoder.h"

void NLPBatchCreator::concatenate_tensors
(torch::Tensor& out)
{
	std::vector<torch::Tensor> list;
	list.push_back(out);
	torch::TensorList tensorlist = torch::TensorList(list);
	out = torch::pad_sequence(tensorlist);
}

void NLPBatchCreator::to_one_hot_based_tensor
(torch::Tensor& out, std::string surname, int padding)
{
	OHVCoder::line_to_one_hot_tensor(out, surname, padding);
	concatenate_tensors(out);
	//std::cout << out.sizes() << std::endl;
}

std::string NLPBatchCreator::back_to_string(torch::Tensor& const in)
{
    extern char all_characters[];

    std::string str = "";
    auto sequence = in.size(0);
    for (int i = 0; i < sequence; i++) {
        int j = 0;
        for (; j < strlen(all_characters); j++) {
            auto val = in.index({ i, 0, j }).item();
            if (val.toFloat() == 1.f) {
                break;
            }
        }

        str += all_characters[j];
    }

    return str;
}