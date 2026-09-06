// Copyright (C) 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <vart/runner_ext.hpp>
#include <vart/tensor_buffer.hpp>

namespace GraphEngine
{
  /**
   * Create a vector of inputs (Space is allocated), (Space is owned by Client)
   *
   * @param runner
   *  Unique Ptr to the runner.
   * @return
   *  Vector of client owned tensor buffers where first element corresponds to the first input, and so on.
   *
   * Use this API if you want to allocate your own input buffers.
   * This API guarantees that they will have been created with the properties required by the hw accelerator.
   * These properties are things such as memory bank assignment, alignment, padding, etc...
   *
   */
  std::vector<std::unique_ptr<vart::TensorBuffer>> create_inputs(std::unique_ptr<vart::RunnerExt> &runner);

  /**
   * Create a vector of inputs (Space is allocated), (Space is owned by Client)
   *
   * @param runner
   *  Raw Ptr to the runner.
   * @return
   *  Vector of client owned tensor buffers where first element corresponds to the first input, and so on.
   *
   * Use this API if you want to allocate your own input buffers.
   * This API guarantees that they will have been created with the properties required by the hw accelerator.
   * These properties are things such as memory bank assignment, alignment, padding, etc...
   *
   */
  std::vector<std::unique_ptr<vart::TensorBuffer>> create_inputs(vart::RunnerExt *runner);

  /**
   * Create a vector of outputs (Space is allocated), (Space is owned by Client)
   *
   * @param runner
   *  Unique Ptr to the runner.
   * @return
   *  Vector of client owned tensor buffers where first element corresponds to the first output, and so on.
   *
   * Use this API if you want to allocate your own output buffers.
   * This API guarantees that they will have been created with the properties required by the hw accelerator.
   * These properties are things such as memory bank assignment, alignment, padding, etc...
   *
   */
  std::vector<std::unique_ptr<vart::TensorBuffer>> create_outputs(std::unique_ptr<vart::RunnerExt> &runner);

  /**
   * Create a vector of outputs (Space is allocated), (Space is owned by Client)
   *
   * @param runner
   *  Raw Ptr to the runner.
   * @return
   *  Vector of client owned tensor buffers where first element corresponds to the first output, and so on.
   *
   * Use this API if you want to allocate your own output buffers.
   * This API guarantees that they will have been created with the properties required by the hw accelerator.
   * These properties are things such as memory bank assignment, alignment, padding, etc...
   *
   */
  std::vector<std::unique_ptr<vart::TensorBuffer>> create_outputs(vart::RunnerExt *runner);

  /**
   * Create a vector of inputs (Space is allocated), (Space is owned by Client)
   *
   * @param runner
   *  Unique Ptr to the dpu runner.
   * @return
   *  Vector of client owned tensor buffers where first element corresponds to the first input, and so on.
   *
   * Use this API if you want to allocate your own input buffers.
   * This API guarantees that they will have been created with the properties required by the hw accelerator.
   * These properties are things such as memory bank assignment, alignment, padding, etc...
   *
   */
  std::vector<std::unique_ptr<vart::TensorBuffer>> create_inputs(std::unique_ptr<vart::Runner> &runner);

  /**
   * Create a vector of inputs (Space is allocated), (Space is owned by Client)
   *
   * @param runner
   *  Raw Ptr to the dpu runner.
   * @return
   *  Vector of client owned tensor buffers where first element corresponds to the first input, and so on.
   *
   * Use this API if you want to allocate your own input buffers.
   * This API guarantees that they will have been created with the properties required by the hw accelerator.
   * These properties are things such as memory bank assignment, alignment, padding, etc...
   *
   */
  std::vector<std::unique_ptr<vart::TensorBuffer>> create_inputs(vart::Runner *runner);

  /**
   * Create a vector of outputs (Space is allocated), (Space is owned by Client)
   *
   * @param runner
   *  Unique Ptr to the dpu runner.
   * @return
   *  Vector of client owned tensor buffers where first element corresponds to the first output, and so on.
   *
   * Use this API if you want to allocate your own output buffers.
   * This API guarantees that they will have been created with the properties required by the hw accelerator.
   * These properties are things such as memory bank assignment, alignment, padding, etc...
   *
   */
  std::vector<std::unique_ptr<vart::TensorBuffer>> create_outputs(std::unique_ptr<vart::Runner> &runner);

  /**
   * Create a vector of outputs (Space is allocated), (Space is owned by Client)
   *
   * @param runner
   *  Raw Ptr to the dpu runner.
   * @return
   *  Vector of client owned tensor buffers where first element corresponds to the first output, and so on.
   *
   * Use this API if you want to allocate your own output buffers.
   * This API guarantees that they will have been created with the properties required by the hw accelerator.
   * These properties are things such as memory bank assignment, alignment, padding, etc...
   *
   */
  std::vector<std::unique_ptr<vart::TensorBuffer>> create_outputs(vart::Runner *runner);

  /**
   * use xir::Attrs to pass info to RunnerExt at runtime.
   * return 0: good; 1: error, RunnerExt cannot find any useful info in Attrs
   */
  int set_run_attrs(std::unique_ptr<vart::RunnerExt>& runner, std::unique_ptr<xir::Attrs>&);

  /**
   * Get the scaling factors for each input.
   *
   * @param runner
   *  Unique Ptr to the runner.
   * @return
   *  Vector of scaling factors, where first element corresponds to the first input, and so on.
   *
   *  Typically XMODELs have integer inputs, and if this is true, each input will be associated with a scaling factor.
   *  This scaling factor is determined by a quantization process, and the metadata is embedded in the XMODEL.
   *  Converting from float to int, involves multiplying the float by this scaling factor, and rounding to the nearest integer.
   *  You can use this API if you need to do the conversion yourself.
   */
  std::vector<float> get_input_scale_factors(std::unique_ptr<vart::RunnerExt> &runner);

  /**
   * Get the scaling factors for each input.
   *
   * @param runner
   *  Pointer to the runner.
   * @return
   *  Vector of scaling factors, where first element corresponds to the first input, and so on.
   *
   *  Typically XMODELs have integer inputs, and if this is true, each input will be associated with a scaling factor.
   *  This scaling factor is determined by a quantization process, and the metadata is embedded in the XMODEL.
   *  Converting from float to int, involves multiplying the float by this scaling factor, and rounding to the nearest integer.
   *  You can use this API if you need to do the conversion yourself.
   */
  std::vector<float> get_input_scale_factors(vart::RunnerExt *runner);

  /**
   * Get the scaling factors for each output.
   *
   * @param runner
   *  Unique Ptr to the runner.
   * @return
   *  Vector of scaling factors, where first element corresponds to the first output, and so on.
   *
   *  Typically XMODELs have integer outputs, and if this is true, each output will be associated with a scaling factor.
   *  This scaling factor is determined by a quantization process, and the metadata is embedded in the XMODEL.
   *  Converting from int to float, involves multiplying the int by this scaling factor
   *  You can use this API if you need to do the conversion yourself.
   */
  std::vector<float> get_output_scale_factors(std::unique_ptr<vart::RunnerExt> &runner);

  /**
   * Get the scaling factors for each output.
   *
   * @param runner
   *  Pointer to the runner.
   * @return
   *  Vector of scaling factors, where first element corresponds to the first output, and so on.
   *
   *  Typically XMODELs have integer outputs, and if this is true, each output will be associated with a scaling factor.
   *  This scaling factor is determined by a quantization process, and the metadata is embedded in the XMODEL.
   *  Converting from int to float, involves multiplying the int by this scaling factor
   *  You can use this API if you need to do the conversion yourself.
   */
  std::vector<float> get_output_scale_factors(vart::RunnerExt *runner);

  /**
   * Copy a buffer from dataSrc to a TensorBuffer.
   *
   * @param tb
   *  Pointer to the tensor buffer where data will be copied to.
   * @param dataSrc
   *  Vector of int8 to be copied
   *
   * This overload of copy buffer is for convienience, and purely calls memcpy underneath.
   * The tb must have a buffer of type XINT. So this is integer <-> integer copy.
   */
  void copy_buffer(vart::TensorBuffer *tb, std::vector<std::int8_t> &dataSrc);

  /**
   * Copy a buffer from dataSrc to a TensorBuffer.
   *
   * @param tb
   *  Pointer to the tensor buffer where data will be copied to.
   * @param dataSrc
   *  Buffer of int8 to be copied
   * @param srcElements
   *  The number of elements to be copied from the source
   *
   * This overload of copy buffer is for convienience, and purely calls memcpy underneath.
   * The tb must have a buffer of type XINT. So this is integer <-> integer copy.
   */
  void copy_buffer(vart::TensorBuffer *tb, std::int8_t *dataSrc, size_t srcElements);

  /**
   * Copy a buffer from dataSrc to a TensorBuffer.
   *
   * @param tb
   *  Pointer to the tensor buffer where data will be copied to.
   * @param dataSrc
   *  Vector of float to be copied
   *
   * This overload of copy buffer supports two cases.
   * 1. Float -> Float, pure memcpy
   * 2. Float -> Int8, scaling will be performed for you. If the model does not have a scaling factor, this call is invalid, and the application will terminate.
   */
  void copy_buffer(vart::TensorBuffer *tb, std::vector<float> &dataSrc);

  /**
   * Copy a buffer from dataSrc to a TensorBuffer.
   *
   * @param tb
   *  Pointer to the tensor buffer where data will be copied to.
   * @param dataSrc
   *  Buffer of int8 to be copied
   * @param srcElements
   *  The number of elements to be copied from the source
   *
   * This overload of copy buffer supports two cases.
   * 1. Float -> Float, pure memcpy
   * 2. Float -> Int8, scaling will be performed for you. If the model does not have a scaling factor, this call is invalid, and the application will terminate.
   */
  void copy_buffer(vart::TensorBuffer *tb, float *dataSrc, size_t srcElements);

  /**
   * Copy a buffer from TensorBuffer to a dataDst.
   *
   * @param dataDst
   *  Vector of int8 to be filled with data. The vector must have been sized appropriately, before making this call.
   * @param tb
   *  Pointer to the tensor buffer where data will be copied from.
   *
   * This overload of copy buffer is for convienience, and purely calls memcpy underneath.
   * The tb must have a buffer of type XINT. So this is integer <-> integer copy.
   */
  void copy_buffer(std::vector<std::int8_t> &dataDst, vart::TensorBuffer *tb);

  /**
   * Copy a buffer from TensorBuffer to a dataDst.
   *
   * @param dataDst
   *  Buffer of int8 to be filled with data. The vector must have been sized appropriately, before making this call.
   * @param dstElements
   *  The number of elements to be copied to the destination
   * @param tb
   *  Pointer to the tensor buffer where data will be copied from.
   *
   * This overload of copy buffer is for convienience, and purely calls memcpy underneath.
   * The tb must have a buffer of type XINT. So this is integer <-> integer copy.
   */
  void copy_buffer(std::int8_t *dataDst, size_t dstElements, vart::TensorBuffer *tb);

  /**
   * Copy a buffer from TensorBuffer to a dataDst.
   *
   * @param dataDst
   *  Vector of float to be filled with data. The vector must have been sized appropriately, before making this call.
   * @param tb
   *  Pointer to the tensor buffer where data will be copied from.
   *
   * This overload of copy buffer supports two cases.
   * 1. Float -> Float, pure memcpy
   * 2. Int8 -> Float, scaling will be performed for you. If the model does not have a scaling factor, this call is invalid, and the application will terminate.
   */
  void copy_buffer(std::vector<float> &dataDst, vart::TensorBuffer *tb);

  /**
   * Copy a buffer from TensorBuffer to a dataDst.
   *
   * @param dataDst
   *  Buffer of float to be filled with data. The vector must have been sized appropriately, before making this call.
   * @param dstElements
   *  The number of elements to be copied to the destination
   * @param tb
   *  Pointer to the tensor buffer where data will be copied from.
   *
   * This overload of copy buffer supports two cases.
   * 1. Float -> Float, pure memcpy
   * 2. Int8 -> Float, scaling will be performed for you. If the model does not have a scaling factor, this call is invalid, and the application will terminate.
   */
  void copy_buffer(float *dataDst, size_t dstElements, vart::TensorBuffer *tb);

  void copy_buffer(vart::TensorBuffer *tb, std::vector<int16_t> &dataSrc);
  void copy_buffer(vart::TensorBuffer *tb, int16_t *dataSrc, size_t srcElements);
  void copy_buffer(std::vector<int16_t> &dataDst, vart::TensorBuffer *tb);
  void copy_buffer(int16_t *dataDst, size_t dstElements, vart::TensorBuffer *tb);

  // Future can add API to operate on vectors of buffers

}
