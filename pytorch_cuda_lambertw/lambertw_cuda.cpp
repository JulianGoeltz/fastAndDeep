#include <torch/extension.h>

#include <ctime>
#include <fstream>
#include <iostream>

torch::Tensor lambertw0_cuda(torch::Tensor z);

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor lambertw0(torch::Tensor z) {
    /*
     * std::fstream fs;
     * fs.open("/tmp/lambertw_cuda.log", std::fstream::out | std::fstream::app);
     * 
     * {
     *     auto time = std::time(nullptr);
     *     fs << "[" << std::asctime(std::localtime(&time)) << "] Checking input\n"
     *        << std::flush;
     * }
     */

    CHECK_INPUT(z);

    /*
     * {
     *     auto time = std::time(nullptr);
     *     fs << "[" <<  std::asctime(std::localtime(&time)) << "] Computing
     * lambertw..\n" << std::flush;
     * }
     */

    auto retval = lambertw0_cuda(z);

    /*
     * {
     *     auto time = std::time(nullptr);
     *     fs << "[" << std::asctime(std::localtime(&time)) << "] Returning
     * results..\n" << std::flush;
     * }
     * fs.close();
     */

    return retval;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lambertw0", &lambertw0,
          "LambertW (0-branch) in CUDA (invalid input mapped to -inf)");
}
