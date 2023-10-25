

#include <torch/extension.h>
#include <pybind11/pybind11.h>



torch::Tensor cmul_element(const torch::Tensor &x, const torch::Tensor &y);
torch::Tensor cnnl_mm(const torch::Tensor &self, const torch::Tensor &other, 
                    float self_scale, float other_scale);
torch::Tensor cmatmul(const torch::Tensor &a, const torch::Tensor &b,
                    float a_scale, float b_scale);




