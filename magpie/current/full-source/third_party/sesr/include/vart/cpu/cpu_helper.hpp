/*

 * Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 * Copyright (C) 2022 – 2025 Advanced Micro Devices, Inc. All rights reserved.

 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cpu_base_inc.hpp"
#include "cpu_types.hpp"

namespace vart {
namespace cpu {
long get_cpu_num();
void set_cpu_num(int cpu_num_param);
}  // namespace cpu
}  // namespace vart

#define CPU_NUM (vart::cpu::get_cpu_num())
#define CPU_RUN_ID "cpu_run_id"

#define PRINT_DIVIDING_LINE()                                                  \
  if (VART_DEBUG) {                                                            \
    UNI_LOG_DEBUG_INFO << "----------------------------------------" << endl;  \
  }

#define INSTANTIATE_TPCLASS(cpu_op_type)                                       \
  template class cpu_op_type<int32_t>;                                         \
  template class cpu_op_type<float>;                                           \
  template class cpu_op_type<double>;

#define INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(cpu_op_type)                     \
  template class cpu_op_type<float, float>;                                    \
  template class cpu_op_type<float, double>;                                   \
  template class cpu_op_type<float, int32_t>;                                  \
  template class cpu_op_type<double, float>;                                   \
  template class cpu_op_type<double, double>;                                  \
  template class cpu_op_type<double, int32_t>;                                 \
  template class cpu_op_type<int32_t, float>;                                  \
  template class cpu_op_type<int32_t, double>;                                 \
  template class cpu_op_type<int32_t, int32_t>;

#define INSTANTIATE_TPCLASS_BINARY(cpu_op_type)                                \
  template class cpu_op_type<float, float, float>;                             \
  template class cpu_op_type<double, float, double>;                           \
  template class cpu_op_type<float, double, double>;                           \
  template class cpu_op_type<double, double, double>;                          \
  template class cpu_op_type<int32_t, int32_t, int32_t>;                       \
  template class cpu_op_type<int32_t, float, float>;                           \
  template class cpu_op_type<float, int32_t, float>;                           \
  template class cpu_op_type<int32_t, double, double>;                         \
  template class cpu_op_type<double, int32_t, double>;

#define INSTANTIATE_TPCLASS_SPECIFIED(cpu_op_type)                             \
  template class cpu_op_type<float, float>;                                    \
  template class cpu_op_type<float, double>;                                   \
  template class cpu_op_type<float, int32_t>;                                  \
  template class cpu_op_type<double, float>;                                   \
  template class cpu_op_type<double, double>;                                  \
  template class cpu_op_type<double, int32_t>;                                 \
  template class cpu_op_type<int32_t, float>;                                  \
  template class cpu_op_type<int32_t, double>;                                 \
  template class cpu_op_type<int32_t, int32_t>;

#define INSTANTIATE_TPCLASS_MATMUL(cpu_op_type)                                \
  INSTANTIATE_TPCLASS_BINARY(cpu_op_type)

#define VART_BIG_THREE_LAW(T)                                                  \
  ~T() = default;                                                              \
  T(const T&) = default;                                                       \
  T& operator=(const T&) = default;

#define VART_DISABLE_COPY_AND_ASSIGN(T)                                        \
  T(const T&) = delete;                                                        \
  T& operator=(const T&) = delete;                                             \
  T(T&&) = delete;                                                             \
  T& operator=(T&&) = delete;

#define VART_DISABLE_COPY(T)                                                   \
  T(const T&) = delete;                                                        \
  T& operator=(const T&) = delete;

#define VART_DISABLE_MOVE(T)                                                   \
  T(T&&) = delete;                                                             \
  T& operator=(T&&) = delete;

#define GET_INPUT_DIM1_FMAP(fmap, input_type)                                  \
  do {                                                                         \
    auto* xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor(#input_type, 0); \
    UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);                     \
    auto xir_tensor_i_dims = xir_tensor_i->get_shape();                        \
    UNI_LOG_CHECK(xir_tensor_i_dims.size() == 1, VART_SIZE_ERROR)              \
        << ", " << xir_tensor_i_dims.size() << " != 1";                        \
    fmap = FMap_t{xir_tensor_i_dims};                                          \
  } while (0)

#define GET_INPUT_DIM4_FMAP(fmap, input_type)                                  \
  do {                                                                         \
    auto* xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor(#input_type, 0); \
    UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);                     \
    auto xir_tensor_i_dims = xir_tensor_i->get_shape();                        \
    UNI_LOG_CHECK(xir_tensor_i_dims.size() == 4, VART_SIZE_ERROR)              \
        << ", " << xir_tensor_i_dims.size() << " != 4";                        \
    fmap = FMap_t{xir_tensor_i_dims};                                          \
  } while (0)

#define GET_INPUT_DIM4_FMAPS(fmap, input_type)                                 \
  do {                                                                         \
    auto input_num = CPUOPBase::xir_op_->get_input_num(#input_type);           \
    UNI_LOG_CHECK(input_num >= 1, VART_SIZE_ERROR)                             \
        << ", " << input_num << " < 1";                                        \
    fmap.clear();                                                              \
    for (auto i = 0; i < input_num; i++) {                                     \
      auto* xir_tensor_i =                                                     \
          CPUOPBase::xir_op_->get_input_tensor(#input_type, i);                \
      UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);                   \
      auto xir_tensor_i_dims = xir_tensor_i->get_shape();                      \
      UNI_LOG_CHECK(xir_tensor_i_dims.size() == 4, VART_SIZE_ERROR)            \
          << ", " << xir_tensor_i_dims.size() << " != 4";                      \
      fmap.emplace_back(xir_tensor_i_dims);                                    \
    }                                                                          \
  } while (0)

#define GET_OUTPUT_DIM4_FMAP(fmap)                                             \
  do {                                                                         \
    auto* xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();              \
    UNI_LOG_CHECK(xir_tensor_o != nullptr, VART_NULL_PTR);                     \
    auto xir_tensor_o_dims = xir_tensor_o->get_shape();                        \
    UNI_LOG_CHECK(xir_tensor_o_dims.size() == 4, VART_SIZE_ERROR)              \
        << ", " << xir_tensor_o_dims.size() << " != 4";                        \
    fmap = FMap_t{xir_tensor_o_dims};                                          \
  } while (0)

#define GET_INPUT_DIMX_FMAP(fmap, input_type, max_dim)                         \
  do {                                                                         \
    auto* xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor(#input_type, 0); \
    UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);                     \
    auto xir_tensor_i_dims = xir_tensor_i->get_shape();                        \
    UNI_LOG_CHECK(xir_tensor_i_dims.size() <= max_dim, VART_SIZE_ERROR)        \
        << ", " << xir_tensor_i_dims.size() << " > max_dim";                   \
    fmap = xir_tensor_i_dims;                                                  \
  } while (0)

#define GET_INPUT_ANY_DIMX_FMAP(fmap, input_type)                              \
  do {                                                                         \
    auto* xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor(#input_type, 0); \
    UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);                     \
    auto xir_tensor_i_dims = xir_tensor_i->get_shape();                        \
    fmap = xir_tensor_i_dims;                                                  \
  } while (0)

#define GET_INPUT_DIMX_FMAPS(fmap, input_type, max_dim)                        \
  do {                                                                         \
    auto input_num = CPUOPBase::xir_op_->get_input_num(#input_type);           \
    UNI_LOG_CHECK(input_num >= 1, VART_SIZE_ERROR)                             \
        << ", " << input_num << " < 1";                                        \
    fmap.clear();                                                              \
    for (auto i = 0; i < input_num; i++) {                                     \
      auto* xir_tensor_i =                                                     \
          CPUOPBase::xir_op_->get_input_tensor(#input_type, i);                \
      UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);                   \
      auto xir_tensor_i_dims = xir_tensor_i->get_shape();                      \
      UNI_LOG_CHECK(xir_tensor_i_dims.size() <= max_dim, VART_SIZE_ERROR)      \
          << ", " << xir_tensor_i_dims.size() << " > max_dim";                 \
      fmap.emplace_back(xir_tensor_i_dims);                                    \
    }                                                                          \
  } while (0)

#define GET_INPUT_ANY_DIM_FMAPS(fmap, input_type)                              \
  do {                                                                         \
    auto input_num = CPUOPBase::xir_op_->get_input_num(#input_type);           \
    UNI_LOG_CHECK(input_num >= 1, VART_SIZE_ERROR)                             \
        << ", " << input_num << " < 1";                                        \
    fmap.clear();                                                              \
    for (auto i = 0; i < input_num; i++) {                                     \
      auto* xir_tensor_i =                                                     \
          CPUOPBase::xir_op_->get_input_tensor(#input_type, i);                \
      UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);                   \
      auto xir_tensor_i_dims = xir_tensor_i->get_shape();                      \
      fmap.emplace_back(xir_tensor_i_dims);                                    \
    }                                                                          \
  } while (0)

#define GET_OUTPUT_DIMX_FMAP(fmap, max_dim)                                    \
  do {                                                                         \
    auto* xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();              \
    UNI_LOG_CHECK(xir_tensor_o != nullptr, VART_NULL_PTR);                     \
    auto xir_tensor_o_dims = xir_tensor_o->get_shape();                        \
    UNI_LOG_CHECK(xir_tensor_o_dims.size() <= max_dim, VART_SIZE_ERROR)        \
        << ", " << xir_tensor_o_dims.size() << " != max_dim";                  \
    fmap = xir_tensor_o_dims;                                                  \
  } while (0)

#define GET_OUTPUT_ANY_DIM_FMAP(fmap)                                          \
  do {                                                                         \
    auto* xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();              \
    UNI_LOG_CHECK(xir_tensor_o != nullptr, VART_NULL_PTR);                     \
    auto xir_tensor_o_dims = xir_tensor_o->get_shape();                        \
    fmap = xir_tensor_o_dims;                                                  \
  } while (0)

namespace vart {
namespace cpu {

class CPUTensorBuffer;

char* TBPTR(TensorBuffer* tb);
size_t TBSIZE(TensorBuffer* tb);

// op related
vector<xir::Op*> get_input_ops(const xir::Subgraph* subg);
vector<xir::Op*> get_output_ops(const xir::Subgraph* subg);
map<string, vector<const xir::Op*>> get_input_map(const xir::Op* op);

// tensor related
bool get_if_signed(const xir::DataType::Type& dtype);
std::string get_data_type_str(const xir::Tensor* xir_tensor);

// print function
void print_xir_op(const xir::Op* xir_op);
void print_cpu_tb(const CPUTensorBuffer* cputb);
void print_xir_subg(const xir::Subgraph* xir_subg);

// pad related
Pad_t calc_pad(FMap_t fmap_i, FMap_t fmap_o, Kernel_t kernel, Stride_t stride,
               Pad_t pad);

template <typename T>
T GET_OP_ATTR(const xir::Op* xir_op, const string& attr) {
  if (!xir_op->has_attr(attr)) {
    UNI_LOG_FATAL(VART_NOT_FOUND) << "Not found attr " << attr << endl;
  }

  T val = xir_op->get_attr<T>(attr);
  return val;
}

template <typename T>
T GET_OP_ATTR_WITH_DFT(const xir::Op* xir_op, const string& attr, T dft) {
  if (!xir_op->has_attr(attr)) {
    if (VART_DEBUG) {
      UNI_LOG_DEBUG_INFO << "Not found attr " << attr
                         << ", use default value instead!" << endl;
    }
    return dft;
  }

  T val = xir_op->get_attr<T>(attr);
  return val;
}

template <typename T>
T GET_INPUT_TENSOR_ATTR(const xir::Op* xir_op, const string& tensor_name,
                        const string& attr) {
  auto* tensor = xir_op->get_input_tensor(tensor_name);
  UNI_LOG_CHECK(tensor != nullptr, VART_NULL_PTR);

  if (!tensor->has_attr(attr)) {
    UNI_LOG_FATAL(VART_NOT_FOUND) << "Not found attr " << attr << endl;
  }

  T val = tensor->get_attr<T>(attr);
  return val;
}

template <typename T>
T GET_INPUT_TENSOR_ATTR_WITH_DFT(const xir::Op* xir_op,
                                 const string& tensor_name, const string& attr,
                                 T dft) {
  auto* tensor = xir_op->get_input_tensor(tensor_name);
  UNI_LOG_CHECK(tensor != nullptr, VART_NULL_PTR);

  if (!tensor->has_attr(attr)) {
    UNI_LOG_DEBUG_INFO << "Not found attr " << attr
                       << ", use default value instead!" << endl;
    return dft;
  }

  T val = tensor->get_attr<T>(attr);
  return val;
}

template <typename T>
T GET_OUTPUT_TENSOR_ATTR(const xir::Op* xir_op, const string& attr) {
  auto* tensor = xir_op->get_output_tensor();
  UNI_LOG_CHECK(tensor != nullptr, VART_NULL_PTR);

  if (!tensor->has_attr(attr)) {
    UNI_LOG_FATAL(VART_NOT_FOUND) << "Not found attr " << attr << endl;
  }

  T val = tensor->get_attr<T>(attr);
  return val;
}

template <typename T>
T GET_OUTPUT_TENSOR_ATTR_WITH_DFT(const xir::Op* xir_op, const string& attr,
                                  T dft) {
  auto* tensor = xir_op->get_output_tensor();
  UNI_LOG_CHECK(tensor != nullptr, VART_NULL_PTR);

  if (!tensor->has_attr(attr)) {
    UNI_LOG_DEBUG_INFO << "Not found attr " << attr
                       << ", use default value instead!" << endl;
    return dft;
  }

  T val = tensor->get_attr<T>(attr);
  return val;
}

}  // namespace cpu
}  // namespace vart
