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

#include <numeric>

#include "cpu_base_inc.hpp"
#include "cpu_op_visitor.hpp"
#include "cpu_reg_func.hpp"
#include "cpu_tensor_buffer.hpp"

namespace vart {
namespace cpu {
float f_to_bf_aie2p(float data);
float max_aie2p(const float a, const float b );
bool check_signed_zero(const float a, bool sign);

class CPUOPBase {
 public:
  // NOTE: subg_ops is updated when op is running which maybe parallel,
  // so it need use atomic to guarantee thread-safe
  static std::atomic<uint64_t> subg_ops;

  // TODO: this two vars are used to write diff script, will removed later
  const static string SUBG_DIFF_SCRIPT;
  const static string SUBG_DIFF_SCRIPT_HEADER;

  // used to init static vars
  static void StaticInit();

 public:
  explicit CPUOPBase(const xir::Subgraph* subg, const xir::Op* op,
                     IMapTBs_t inputs, CPUTBPtr_t output);
  virtual ~CPUOPBase() = default;
  VART_DISABLE_COPY_AND_ASSIGN(CPUOPBase);

  // virtual interfaces
 public:
  virtual void accept(CPUOPVisitor* v) { v->run(this); }
  virtual void run() = 0;

  virtual void print_param() {}
  virtual void check_param() {}

  virtual void read() {}
  virtual void save();

  virtual uint64_t get_workload() { return 0; }

  // useful routines to get op basic information
 public:
  const xir::Subgraph* get_xir_subg() const { return xir_subg_; }
  const xir::Op* get_xir_op() const { return xir_op_; }
  const xir::Tensor* get_xir_tensor() const { return output_tensor_; }

  string get_name() const { return xir_op_->get_name(); }
  string get_type() const { return xir_op_->get_type(); }
  int get_input_num() const { return xir_op_->get_input_num(); }
  string get_input_list() const;
  string get_output_tensor_name() const { return output_tensor_->get_name(); }
  string get_data_type() const { return get_data_type_str(output_tensor_); }

 protected:
  const xir::Subgraph* xir_subg_{nullptr};
  const xir::Op* xir_op_{nullptr};
  const xir::Tensor* output_tensor_{nullptr};
  IMapTBs_t inputs_;
  CPUTBPtr_t output_;

  xir::DataType::Type data_type_;
  int bit_width_;
  bool if_signed_;
  int32_t data_min_;
  int32_t data_max_;
  string round_mode_;

  uint64_t ops_{0};
};

// Returns a value of type T by reinterpretting the representation of the input
// value (part of C++20).
//
// Provides a safe implementation of type punning.
//
// Constraints:
// - U and T must have the same size
// - U and T must be trivially copyable
template <typename T, typename U>
inline T bit_cast(const U &u) {
    T t;
    // Since bit_cast is used in SYCL kernels it cannot use std::memcpy as it
    // can be implemented as @llvm.objectsize.* + __memcpy_chk for Release
    // builds which cannot be translated to SPIR-V.
    uint8_t *t_ptr = reinterpret_cast<uint8_t *>(&t);
    const uint8_t *u_ptr = reinterpret_cast<const uint8_t *>(&u);
    for (size_t i = 0; i < sizeof(U); i++)
        t_ptr[i] = u_ptr[i];
    return t;
}
}  // namespace cpu
}  // namespace vart
