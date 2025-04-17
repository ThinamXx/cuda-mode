#include <torch/extension.h>

void cudaSoftmax(float *input, float *output, int seq_len);
torch::Tensor softmax(torch::Tensor input) {
    torch::Tensor output = torch::zeros_like(input);
    const int seq_len = input.size(1);
    cudaSoftmax(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        seq_len
    );
    return output;
}

void cudaSoftmaxReduce(float *input, float *output, int seq_len);
torch::Tensor softmax_reduce(torch::Tensor input) {
    torch::Tensor output = torch::zeros_like(input);
    const int seq_len = input.size(1);
    cudaSoftmaxReduce(
        input.data_ptr<float>(), output.data_ptr<float>(), seq_len);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax, "Softmax CUDA");
    m.def("softmax_reduce", &softmax_reduce, "Softmax Reduce CUDA");
}