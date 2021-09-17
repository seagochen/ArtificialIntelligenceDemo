#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>

#pragma comment(lib, "asmjit.lib")
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "fbgemm.lib")
#pragma comment(lib, "torch_cpu.lib")

#include "NLPBatchCreator.h"
#include "Languages.h"

Languages lang;

void test(torch::jit::script::Module& module, std::string name)
{
    // no gradient
    torch::NoGradGuard no_grad;

    // convert string to tensor
    torch::Tensor ts;
    NLPBatchCreator::to_one_hot_based_tensor(ts, name);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(ts);

    // forward computation
    auto probabilities = module.forward(inputs).toTensor();

    // print out predication
    auto tp = torch::max(probabilities, 1); // dim = 1
    auto max_id = std::get<1>(tp).item().toInt();

    std::cout << "Is this name " << lang[max_id] << "?" << std::endl;

    // print out each probabilities
    lang.lang_with_perhaps(probabilities);
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model." << std::endl;
        return -1;
    }

    // test input
    std::string input;
    while (true) {
        std::cout << "Name or exit?" << std::endl;
        std::cin >> input;

        if (input == "exit") {
            break;
        }

        // test input
        test(module, input);
    }

    std::cout << "Adios~!" << std::endl;
    return 0;
}
