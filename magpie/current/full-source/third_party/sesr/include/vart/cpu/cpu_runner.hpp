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

#include <unordered_map>
#include "cpu_base_inc.hpp"

namespace vart {
namespace cpu {

class CPUTBFactory;
class CPUOPBase;

class CPURunner : public RunnerExt {
 public:
  using OpRunInputMap = map<string, vector<shared_ptr<TensorBuffer>>>;
  using OpRunFunc = function<void(OpRunInputMap, shared_ptr<TensorBuffer>)>;

 public:
  explicit CPURunner(const xir::Subgraph* subgraph,
                     const xir::Attrs* attrs = nullptr);
  virtual ~CPURunner();

  CPURunner(const CPURunner&) = delete;
  CPURunner& operator=(const CPURunner&) = delete;
  CPURunner(CPURunner&&) = delete;
  CPURunner& operator=(CPURunner&&) = delete;

 public:
  void create_ops_and_tbs();
  virtual std::pair<uint32_t, int>  // pair<jodid, status>
  execute_async(
      const std::vector<TensorBuffer*>& sbug_input_tbs,
      const std::vector<TensorBuffer*>& subg_output_tbs) override final;
  virtual int wait(int jobid, int ms) override final;

  std::vector<const xir::Tensor*> get_input_tensors() final;
  std::vector<const xir::Tensor*> get_output_tensors() final;

 public:
  vector<TensorBuffer*> get_inputs() final;
  vector<TensorBuffer*> get_outputs() final;
  int set_run_attrs(std::unique_ptr<xir::Attrs>&) final { return 0; }
  vector<TensorBuffer*> get_outer_tensor_buffers(const vector<xir::Op*>& ops);
  vector<TensorBuffer*> get_input_tbs();
  vector<TensorBuffer*> get_output_tbs();
  TensorBuffer* get_tb(const xir::Tensor* tensor);

  void run();
  void set_subg_input_tbs(const std::vector<TensorBuffer*>& sbug_input_tbs);
  void set_subg_output_tbs(const std::vector<TensorBuffer*>& subg_output_tbs);
  void set_subg_input_data();
  void set_subg_output_data();

 public:  // mainly used in python api
  string get_name() const;
  string get_device() const;
  int get_op_num() const;
  int get_op_num(const string& type) const;
  int get_op_type_num() const;
  string get_op_type_list() const;
  list<xir::Op*> get_unregistered_ops();

  vector<CPUOPBase*> get_cpu_ops() const;
  CPUOPBase* get_cpu_op(const xir::Op* op) const;

 private:
  const xir::Subgraph* subg_;
  const xir::Graph* g_;

  std::vector<CPUOPBase*> cpu_ops_;
  std::vector<xir::Op*> xir_ops_;
  mutable std::unordered_map<const xir::Op*, CPUOPBase*> cpu_op_map_;

  std::vector<TensorBuffer*> outer_subg_input_tbs_;
  std::vector<TensorBuffer*> outer_subg_output_tbs_;

  std::vector<std::string> assign_tensors_;
  std::vector<std::string> run_from_tensors_;
  std::future<void> fut_;

  std::unordered_map<const xir::Tensor*, CPUTensorBuffer*> outer_tbs_map_;
};

}  // namespace cpu
}  // namespace vart
