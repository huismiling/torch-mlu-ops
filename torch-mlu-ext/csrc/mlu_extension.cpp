
#include <mlu_extension.h>

#include "ATen/Tensor.h"

#include "aten/utils/types.h"
#include "aten/utils/tensor_util.h"
#include "aten/cnnl/cnnlHandle.h"
#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"

using namespace torch_mlu;

void bang_mul_element(void *a, void *b, void *c, int32_t size,
                    cnrtDim3_t taskDim, cnrtFunctionType_t ktype,
                    cnrtQueue_t queue, cnrtDataType_t dtype);

at::Tensor get_contiguous(const at::Tensor& input,
                           c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous)
{
    auto out = torch_mlu::cnnl_contiguous(input, memory_format);
    return out;
}

torch::Tensor cmul_element(const torch::Tensor &x, const torch::Tensor &y)
{
    auto x_contiguous = get_contiguous(x);
    auto y_contiguous = get_contiguous(y);
    auto x_impl = getMluTensorImpl(x_contiguous);
    auto x_ptr = x_impl->cnnlMalloc();
    auto y_impl = getMluTensorImpl(y_contiguous);
    auto y_ptr = y_impl->cnnlMalloc();

    auto z = at::empty_like(x_contiguous);
    auto z_impl = getMluTensorImpl(z);
    auto z_ptr = z_impl->cnnlMalloc();

    int32_t size = x_contiguous.numel();

    cnrtDataType_t dtype = cnnlType2CnrtType(getCnnlType(x_impl));
    cnrtDim3_t taskDim = {32, 1, 1};
    cnrtFunctionType_t ktype = CNRT_FUNC_TYPE_BLOCK;

    cnrtQueue_t queue = getCurQueue();
    bang_mul_element((void*)x_ptr, (void*)y_ptr, (void*)z_ptr, size,
        taskDim, ktype, queue, dtype);

    return z;
}

torch::Tensor cmatmul(const torch::Tensor &a, const torch::Tensor &b,
                    float a_scale=1.0, float b_scale=1.0)
{
    auto z = cnnl_mm(a, b, a_scale, b_scale);
    return z;
}

PYBIND11_MODULE(libmlu_ext, m)
{
    m.def("cmul_element", &cmul_element);
    m.def("cmatmul", &cmatmul);
}

