/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <algorithm>

#include "ATen/NativeFunctions.h"
#include "ATen/Tensor.h"

#include "aten/util/types.h"
#include "aten/util/tensor_util.h"
#include "aten/cnnl/cnnlHandle.h"
#include "aten/cnnl/cnnl_util.h"
#include "aten/cnnl/cnnlCommonDescriptors.h"
#include "aten/cnnl/cnnlTensorDescriptors.h"
#include "aten/cnnl/cnnlOpDescriptors.h"
#include "aten/cnnl/cnnlAlgorithms.h"
#include "aten/cnnl/cnnlHeuristicResult.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

using namespace torch_mlu;

std::set<at::ScalarType> mm_support_dtype{at::ScalarType::Half,
                                          at::ScalarType::Float,
                                          at::ScalarType::Double,
                                          at::ScalarType::Int,
                                          at::ScalarType::Char,
                                          at::ScalarType::Short};

at::Tensor getMatmulOut(const at::Tensor &self,
                        const at::Tensor &other,
                        bool is_trans_self,
                        bool is_trans_other,
                        at::TensorOptions output_options) {
    auto self_shape = self.sizes();
    auto other_shape = other.sizes();
    std::vector<int64_t> output_shape(2, 1);
    if (is_trans_self) {
        output_shape[0] = self_shape[1];
    } else {
        output_shape[0] = self_shape[0];
    }
    if (is_trans_other) {
        output_shape[1] = other_shape[0];
    } else {
        output_shape[1] = other_shape[1];
    }
    return at::empty(output_shape, output_options);
}

std::tuple<at::Tensor, bool> getMMInput(const at::Tensor &self) {
    TORCH_MLU_CHECK(self.dim() == 2, "dimension must be 2 in mm.");
    bool is_trans_self;
    if ((!self.is_contiguous())
        && (self.is_non_overlapping_and_dense())
        && (self.t().is_contiguous())) {
      is_trans_self = true;
      return std::make_tuple(self.t(), is_trans_self);
    } else {
      is_trans_self = false;
      return std::make_tuple(torch_mlu::cnnl::ops::cnnl_contiguous(self, c10::MemoryFormat::Contiguous), is_trans_self);
    }
}

at::Tensor cnnl_mm_out_internal(at::Tensor &result, const at::Tensor &self,
                                   const at::Tensor &other, const at::Tensor &vec,
                                   bool is_trans_self,
                                   bool is_trans_other, bool is_trans_vec,
                                   const at::Scalar &beta_, const at::Scalar &alpha_,
                                   bool allow_tf32,
                                   float self_scale, float other_scale) {
  TORCH_MLU_CHECK(mm_support_dtype.find(self.scalar_type()) != mm_support_dtype.end(),
                  "MM mlu op not implemented for dtype of input1: '",
                  self.dtype().name(), "'");

  TORCH_MLU_CHECK(mm_support_dtype.find(other.scalar_type()) != mm_support_dtype.end(),
                  "MM mlu op not implemented for dtype of input2: '",
                  other.dtype().name(), "'");

  if (vec.defined()) {
    TORCH_MLU_CHECK(mm_support_dtype.find(vec.scalar_type()) != mm_support_dtype.end(),
                    "MM mlu op not implemented for dtype of input3: '",
                    vec.dtype().name(), "'");
  }

  // get tensor impl
  auto self_impl = getMluTensorImpl(self);
  auto other_impl = getMluTensorImpl(other);
  auto vec_impl = vec.defined() ? getMluTensorImpl(vec) : nullptr;
  auto result_impl = getMluTensorImpl(result);

  // create desc
  CnnlMatmulDescriptor matmul_desc;
  CnnlMatmulAlgorithm matmul_algo;
  cnnlMatMulPrefer_t preference;
  CnnlMatmulHeuristicResult matmul_hr;
  CnnlTensorDescriptor self_desc;
  CnnlTensorDescriptor other_desc;
  CnnlTensorDescriptor vec_desc;
  CnnlTensorDescriptor result_desc;
  int return_algo_count;
  int requested_algo_count = 1;
  int32_t is_trans_self_int = is_trans_self;
  int32_t is_trans_other_int = is_trans_other;
  int32_t is_trans_vec_int = is_trans_vec;
  int32_t allow_tf32_int = allow_tf32 ? 1 : 0;

  matmul_desc.set_attr(CNNL_MATMUL_ALLOW_TF32,
                       &(allow_tf32_int),
                       sizeof(int32_t));
  matmul_desc.set_attr(CNNL_MATMUL_DESC_TRANSA,
                       &(is_trans_self_int),
                       sizeof(int32_t));
  matmul_desc.set_attr(CNNL_MATMUL_DESC_TRANSB,
                       &(is_trans_other_int),
                       sizeof(int32_t));
  if (vec.defined()) {
    int32_t use_beta = true;
    matmul_desc.set_attr(CNNL_MATMUL_USE_BETA,
                         &(use_beta),
                         sizeof(int32_t));
  }
  // TODO(xushuo): CNNL_MATMUL_DESC_TRANSC will be supported in the future
  // matmul_desc.set_attr(CNNL_MATMUL_DESC_TRANSC,
  //                      &(is_trans_vec_int),
  //                      sizeof(int32_t));

  auto self_type = self.dtype();
  auto other_type = other.dtype();

  if (self_type.name() == std::string("int"))
    self_desc.set(self, CNNL_DTYPE_INT31);
  else
    self_desc.set(self);

  if (other_type.name() == std::string("int"))
    other_desc.set(other, CNNL_DTYPE_INT31);
  else
    other_desc.set(other);
  
  int p = std::floor(std::log(1/self_scale) / std::log(2));
  auto s = std::pow(2, p) *self_scale;
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorPositionScaleAndOffset(self_desc.desc(),
                    p, s, 0));
  p = std::floor(std::log(1/other_scale) / std::log(2));
  s = std::pow(2, p) *other_scale;
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorPositionScaleAndOffset(other_desc.desc(),
                    p, s, 0));

  if (vec.defined()) {
    auto vec_type = vec.dtype();
    if (vec_type.name() == std::string("int")) {
      vec_desc.set(vec, CNNL_DTYPE_INT31);
    } else {
      vec_desc.set(vec);
    }
  }
  result_desc.set(result);

  auto handle = getCurrentHandle();

  matmul_hr.get(handle,
              matmul_desc.desc(),
              self_desc.desc(),
              other_desc.desc(),
              vec.defined() ? vec_desc.desc() : result_desc.desc(),
              result_desc.desc(),
              preference,
              requested_algo_count,
              &return_algo_count);

  at::Tensor workspace;
  size_t workspace_size = 0;
  void *workspace_ptr = nullptr;

  TORCH_CNNL_CHECK(cnnlGetMatMulHeuristicResult(matmul_hr.hr(),
                                                matmul_algo.mut_algo(),
                                                &workspace_size));
  workspace = at::empty(workspace_size,
                        self.options().dtype(at::ScalarType::Char));
  workspace_ptr = getMluTensorImpl(workspace)->mlu_data_ptr();
  auto self_ptr = self_impl->mlu_data_ptr();
  auto other_ptr = other_impl->mlu_data_ptr();
  auto vec_ptr = vec.defined() ? vec_impl->mlu_data_ptr() : nullptr;
  auto result_ptr = result_impl->mlu_data_ptr();
  float alpha_float = alpha_.toFloat();
  float beta_float = beta_.toFloat();
  const void * alpha = static_cast<void *>(&alpha_float);
  const void * beta = static_cast<void *>(&beta_float);

  TORCH_CNNL_CHECK(cnnlMatMul_v2(handle,
                                 matmul_desc.desc(),
                                 matmul_algo.algo(),
                                 alpha,
                                 self_desc.desc(),
                                 self_ptr,
                                 other_desc.desc(),
                                 other_ptr,
                                 beta,
                                 vec.defined() ? vec_desc.desc() : result_desc.desc(),
                                 vec_ptr ? vec_ptr : result_ptr,
                                 workspace_ptr,
                                 workspace_size,
                                 result_desc.desc(),
                                 result_ptr));
  auto queue = getCurrentQueue();
  queue.synchronize();

  return result;
}
at::Tensor cnnl_mm_internal(const at::Tensor &self, const at::Tensor &other,
                            at::TensorOptions self_options, bool is_trans_self,
                            bool is_trans_other, bool allow_tf32,
                            float self_scale, float other_scale) {
  // get the shape of output
  TORCH_CHECK(self.dim() == 2 && other.dim() == 2, "dimension not support");

  auto output = getMatmulOut(self, other, is_trans_self, is_trans_other, self_options);
  if (output.numel() == 0)
    return output;

  if (!at::isFloatingType(output.scalar_type())) {
    CNLOG(INFO) << "Output dtype : "
                << output.dtype().name()
                << " is not supported, and will cast to float due to limit of MM op.";
    output = getMatmulOut(self, other, is_trans_self, is_trans_other,
                          self_options.dtype(at::ScalarType::Half)); // Half, Float
  }

  at::Tensor dummy_vec;
  cnnl_mm_out_internal(output, self, other, dummy_vec,
                      is_trans_self, is_trans_other, false, 0.0, 1.0, allow_tf32,
                      self_scale, other_scale);

  return output;
}


at::Tensor cnnl_mm(const at::Tensor &self, const at::Tensor &other, 
                  float self_scale, float other_scale) {
  // TORCH_MLU_CHECK(self.scalar_type() == other.scalar_type(),
  //                 "MM mlu op need self and other be same dtype, but self is : '",
  //                 self.dtype().name(), "'", ", other is : '",
  //                 other.dtype().name(), "'");
  auto mm_arg1 = at::TensorArg(self, "mat1", 1);
  auto mm_arg2 = at::TensorArg(other, "mat2", 1);
  at::checkDim("mm", mm_arg1, 2);
  at::checkDim("mm", mm_arg2, 2);
  TORCH_MLU_CHECK(self.size(1) == other.size(0),
     "size mismatch, m1: ", self.sizes(), ", m2: ", other.sizes(),
     " while checking arguments for mm");
  // case1: self's col and other's row are 0, return zero tensor, size like mm.
  // case2: self's row or other's col is 0, return empty tensor, size like mm.
  // case2 has a higher priority than case1.
  if (self.numel() == 0 || other.numel() == 0) {
    return at::zeros({self.size(0), other.size(1)}, self.options());
  }

  at::Tensor self_contiguous;
  at::Tensor other_contiguous;
  bool is_trans_self;
  bool is_trans_other;
  std::tie(self_contiguous, is_trans_self) = getMMInput(self);
  std::tie(other_contiguous, is_trans_other) = getMMInput(other);

  TORCH_MLU_CHECK(Global::instance().isUsingFloatingDevice(),
                  "MatMul is no longer supported with quantized mode on ",
                  "None-Floating-Point supported device.");
  return cnnl_mm_internal(self_contiguous, other_contiguous,
                          self_contiguous.options(), is_trans_self, is_trans_other,
                          torch_mlu::Global::instance().allowCNNLMatmulTF32(),
                          self_scale, other_scale);
}

