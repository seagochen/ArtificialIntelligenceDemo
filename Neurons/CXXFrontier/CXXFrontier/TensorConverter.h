#pragma once
#ifndef _TENSOR_CONVERTER_H_
#define _TENSOR_CONVERTER_H_

#include <torch/torch.h>
#include <vector>

class TensorConverter
{
public:

	template<typename T>
	static std::vector<T> to(torch::Tensor& const tensor) {
		auto t = tensor.contiguous();
		std::vector<T> v(t.data_ptr<T>(), t.data_ptr<T>() + t.numel());
		return v;
	};
};

#endif