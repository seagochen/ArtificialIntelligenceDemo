#pragma once
#ifndef _OHV_CODER_H_
#define _OHV_CODER_H_

#include <torch/torch.h>
#include <string>

class OHVCoder
{
private:
	static int letter_to_index(char letter);

public:
	static void line_to_one_hot_tensor
	(torch::Tensor& out, std::string line, int max_padding=0);

	static void print_ohv_vector(torch::Tensor& const in);
};

#endif
