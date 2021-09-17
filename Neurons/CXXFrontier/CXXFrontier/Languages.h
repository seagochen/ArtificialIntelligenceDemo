#pragma once
#ifndef _LANGUAGES_H_
#define _LANGUAGES_H_

#include <vector>
#include <string>

#include <torch/torch.h>

class Languages
{
private:
	std::vector<std::string> langs;

	void setup_languages();

public:
	Languages();

	std::string operator[](int indx) const;

	std::string at(int indx) const;
	
	void lang_with_perhaps(torch::Tensor& const in);
};

#endif