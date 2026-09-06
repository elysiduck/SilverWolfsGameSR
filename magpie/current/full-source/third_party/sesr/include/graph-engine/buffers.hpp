// Copyright (C) 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <assert.h>

#include <xir/tensor/tensor.hpp>
#include <vart/tensor_buffer.hpp>

#include <xrt/xrt_bo.h>

/**
 * @class HostBuffer
 *
 * @brief
 * HostBuffer is a derived class of vart::TensorBuffer
 * and it is composed of a std::vector.
 *
 * @details
 * This class exists to comply with the VART APIs.
 * However, it is merely a std::vector. Thus it owns and manages memory.
 */
template <typename T>
class HostBuffer : public vart::TensorBuffer
{

public:
  /**
   * Construct empty HostBuffer object.
   *
   * Can be used as lvalue in assignment.
   */
  HostBuffer() = default;

  /**
   * Destroy HostBuffer object.
   */
  virtual ~HostBuffer() = default;

  /**
   * Construct an instance of a HostBuffer.
   *
   * @param tensor
   *  Pointer to a tensor object that provides shape and datatype information.
   * @param args
   *  Argument list to be forwarded to std::vector constructor.
   */
  template <typename... Args>
  HostBuffer(const xir::Tensor *tensor, Args &&...args)
      : TensorBuffer(tensor), vector_(std::forward<Args>(args)...),
        data_((void *)vector_.data()),
        size_(vector_.size() * sizeof(T)) {}

  /**
   * Get a pair representing the raw data and its size.
   *
   * @param idx
   *  A vector of indexes used to take a slice of the raw data at a given dimension.
   * @return
   *  A pair containing a pointer to the data and its size in bytes.
   */
  std::pair<uint64_t, size_t> data(const std::vector<std::int32_t> idx = {}) override
  {
    assert(idx.size() == 0); // No slicing for you
    return {reinterpret_cast<uint64_t>(data_), size_};
  }

  /**
   * Get a reference to the underlying vector object encapsulated by this TensorBuffer.
   *
   * @return
   *  Reference to the vector contained by this TensorBuffer.
   */
  std::vector<T> &get_vector() { return vector_; }
  
  /**
   * Get a reference to the underlying vector object encapsulated by this TensorBuffer.
   *
   * @param basePtr
   *  A pointer to a vart::TensorBuffer from which we want to extract a vector object.
   * @return
   *  Reference to the vector contained by this TensorBuffer.
   */
  static std::vector<T>& get_vector(vart::TensorBuffer* basePtr) {
    auto *derivedPtr = dynamic_cast<HostBuffer<T>*>(basePtr);
    if(!derivedPtr) // cast failed
      throw std::runtime_error("Error: Failed to get std::vector.");
    return derivedPtr->get_vector();
  }

protected:
  std::vector<T> vector_;
  void *data_;
  size_t size_;
};

/**
 * @class XrtBuffer
 *
 * @brief
 * XrtBuffer is a derived class of vart::TensorBuffer
 * and it is composed of an xrt::bo.
 *
 * @details
 * This class exists to comply with the VART APIs.
 * However, it is merely an xrt::bo. Thus it owns and manages memory.
 */
class XrtBuffer : public vart::TensorBuffer
{
public:
  /**
   * Construct empty HostBuffer object.
   *
   * Can be used as lvalue in assignment.
   */
  XrtBuffer() = default;

  /**
   * Destroy XrtBuffer object.
   */
  virtual ~XrtBuffer() = default;

  /**
   * Construct an instance of a XrtBuffer where there is a relationship to
   * another BO.
   *
   * @param tensor
   *  Pointer to a tensor object that provides shape and datatype information.
   * @param parent
   *  A shared_ptr to xrt::bo object that has parent relationship to this BO.
   * @param strides
   *  Step sizes for each dimension when stepping through this buffer.
   * @param args
   *  Argument list to be forwarded to xrt::bo constructor.
   */
  template <typename... Args>
  XrtBuffer(const xir::Tensor *tensor, std::shared_ptr<xrt::bo> parentPtr, const std::vector<std::int32_t> &strides, Args &&...args)
      : TensorBuffer(tensor),
        parentPtr_(parentPtr),
	bo_(std::forward<Args>(args)...),
        data_(bo_.map<void *>()),
        size_(bo_.size()),
        strides_(strides) {}

  /**
   * Construct an instance of a XrtBuffer.
   *
   * @param tensor
   *  Pointer to a tensor object that provides shape and datatype information.
   * @param strides
   *  Step sizes for each dimension when stepping through this buffer.
   * @param args
   *  Argument list to be forwarded to xrt::bo constructor.
   */
  template <typename... Args>
  XrtBuffer(const xir::Tensor *tensor, const std::vector<std::int32_t> &strides, Args &&...args)
      : TensorBuffer(tensor), bo_(std::forward<Args>(args)...),
        data_(bo_.map<void *>()),
        size_(bo_.size()),
        strides_(strides) {}

  /**
   * Get a pair representing the raw data and its size.
   *
   * @param idx
   *  A vector of indexes used to take a slice of the raw data at a given dimension.
   * @return
   *  A pair containing a pointer to the data and its size in bytes.
   */
  std::pair<uint64_t, size_t> data(const std::vector<std::int32_t> idx = {}) override
  {
    // TODO No slicing for now, this is fine if batch=1
    return {reinterpret_cast<uint64_t>(data_), size_};
  }

  /**
   * Get a reference to the underlying xrt::bo object encapsulated by this TensorBuffer.
   *
   * @return
   *  Reference to the xrt::bo contained by this TensorBuffer.
   */
  xrt::bo &get_bo() { return bo_; }
  
  /**
   * Get a reference to the underlying parent xrt::bo object encapsulated by this TensorBuffer.
   *
   * @return
   *  Reference to the xrt::bo contained by this TensorBuffer.
   */
  xrt::bo &get_parent() { return *parentPtr_; }

  /**
   * Get a reference to the underlying xrt::bo object encapsulated by this TensorBuffer.
   *
   * @param basePtr
   *  A pointer to a vart::TensorBuffer from which we want to extract a BO object.
   * @return
   *  Reference to the xrt::bo contained by this TensorBuffer.
   */
  static xrt::bo& get_bo(vart::TensorBuffer* basePtr) {
    auto *derivedPtr = dynamic_cast<XrtBuffer*>(basePtr);
    if(!derivedPtr) // cast failed
      throw std::runtime_error("Error: Failed to get xrt::bo.");
    return derivedPtr->get_bo();
  }
  
  /**
   * Get a reference to the underlying xrt::bo object encapsulated by this TensorBuffer.
   *
   * @param basePtr
   *  A pointer to a vart::TensorBuffer from which we want to extract a BO object.
   * @return
   *  Reference to the xrt::bo contained by this TensorBuffer.
   */
  static xrt::bo& get_parent(vart::TensorBuffer* basePtr) {
    auto *derivedPtr = dynamic_cast<XrtBuffer*>(basePtr);
    if(!derivedPtr) // cast failed
      throw std::runtime_error("Error: Failed to get xrt::bo.");
    return derivedPtr->get_parent();
  }

  /**
   * Get a reference to the strides information encapsulated by this TensorBuffer.
   *
   * @return
   *  Reference to the strides contained by this TensorBuffer.
   *  these strides are used to walk through the buffer if the data is not contiguous
   */
  const std::vector<std::int32_t>& get_strides() { return strides_; }

  /**
   * Get a reference to the strides information encapsulated by this TensorBuffer.
   *
   * @param basePtr
   *  A pointer to a vart::TensorBuffer from which we want to extract the strides.
   * @return
   *  Reference to the strides contained by this TensorBuffer.
   *  these strides are used to walk through the buffer if the data is not contiguous
   */
  static std::vector<std::int32_t> get_strides(vart::TensorBuffer* basePtr) {
    auto *derivedPtr = dynamic_cast<XrtBuffer*>(basePtr);
    if(!derivedPtr) // cast failed
      return std::vector<std::int32_t>{};
      //throw std::runtime_error("Error: Failed to get strides.");
    return derivedPtr->get_strides();
  }

protected:
  std::shared_ptr<xrt::bo> parentPtr_;
  xrt::bo bo_;
  void *data_;
  size_t size_;
  std::vector<std::int32_t> strides_;
};
