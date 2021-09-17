#pragma once
#ifndef _NLP_BATCH_CREATOR_H_
#define _NLP_BATCH_CREATOR_H_

#include <torch/torch.h>
#include <vector>
#include <string>

class NLPBatchCreator
{
private:
    static void concatenate_tensors
	(torch::Tensor& out);

public:
    static void to_one_hot_based_tensor
    (torch::Tensor& out, std::string surnames, int padding=20);

    static std::string back_to_string(torch::Tensor& const in);
};

#endif