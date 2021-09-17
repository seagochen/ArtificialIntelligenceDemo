#include "Languages.h"
#include <iostream>


void Languages::setup_languages()
{
	langs.push_back("Arabic");
	langs.push_back("Chinese");
	langs.push_back("Czech");
	langs.push_back("Dutch");
	langs.push_back("English");
	langs.push_back("French");
	langs.push_back("German");
	langs.push_back("Greek");
	langs.push_back("Irish");
	langs.push_back("Italian");
	langs.push_back("Japanese");
	langs.push_back("Korean");
	langs.push_back("Polish");
	langs.push_back("Portuguese");
	langs.push_back("Russian");
	langs.push_back("Scottish");
	langs.push_back("Spanish");
	langs.push_back("Vietnamese");
}

Languages::Languages()
{
	setup_languages();
}

std::string Languages::operator[](int indx) const
{
	return langs[indx % langs.size()];
}

std::string Languages::at(int indx) const
{
	return langs[indx % langs.size()];
}

void Languages::lang_with_perhaps(torch::Tensor& const in)
{
	auto count = in.size(1);
	for (int i = 0; i < count; i++) {
		auto scalar = in.index({"...", i}).item();
		std::cout << 
			std::left << std::setw(15) << at(i) << 
			std::left << std::setw(10) << std::setprecision(2) << scalar.toFloat() << std::endl;
	}
}
