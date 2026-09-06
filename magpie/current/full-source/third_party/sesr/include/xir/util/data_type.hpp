/*

 * Copyright (C) 2019 - 2022 Xilinx, Inc. All rights reserved.
 * Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include <climits>
#include <cstdint>
#include <string>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "xir/XirExport.hpp"

namespace xir {

struct XIR_DLLESPEC DataType {
  enum Type { INT, UINT, XINT, XUINT, FLOAT, BFLOAT, UNKNOWN };

  DataType();
  DataType(const std::string& type);
  DataType(const Type& type, const std::int32_t& bit_width);

  const bool valid() const;
  const std::string to_string() const;

  XIR_DLLESPEC friend bool operator==(const DataType& lhs, const DataType& rhs);
  XIR_DLLESPEC friend bool operator!=(const DataType& lhs, const DataType& rhs);

  Type type;
  std::int32_t bit_width;
};

// helper function
template <typename T>
const std::int32_t& get_bit_width() {
  static constexpr std::int32_t bit_width = sizeof(T) * CHAR_BIT;
  return bit_width;
}

template <typename T>
DataType create_data_type() {
  return DataType{"UNKNOWN0"};
};

template <>
XIR_DLLESPEC DataType create_data_type<float>();
template <>
XIR_DLLESPEC DataType create_data_type<double>();
template <>
XIR_DLLESPEC DataType create_data_type<char>();
template <>
XIR_DLLESPEC DataType create_data_type<std::int16_t>();
template <>
XIR_DLLESPEC DataType create_data_type<std::int32_t>();
template <>
XIR_DLLESPEC DataType create_data_type<std::int64_t>();
template <>
XIR_DLLESPEC DataType create_data_type<unsigned char>();
template <>
XIR_DLLESPEC DataType create_data_type<std::uint16_t>();
template <>
XIR_DLLESPEC DataType create_data_type<std::uint32_t>();
template <>
XIR_DLLESPEC DataType create_data_type<std::uint64_t>();

template <typename T, typename U>
inline T bit_cast(const U &u);


struct bfloat16_t;
bool try_cvt_float_to_bfloat16(bfloat16_t *out, const float *inp);

struct bfloat16_t {
  uint16_t raw_bits_;
  bfloat16_t() = default;
  constexpr bfloat16_t(uint16_t r, bool) : raw_bits_(r) {}
  bfloat16_t(float f) { (*this) = f; }

  template <typename IntegerType,
            typename SFINAE = typename std::enable_if<
                std::is_integral<IntegerType>::value>::type>
  bfloat16_t(const IntegerType i)
      : raw_bits_{convert_bits_of_normal_or_zero(
            bit_cast<uint32_t>(static_cast<float>(i)))} {}

  bfloat16_t& operator=(float f) {
    auto iraw = bit_cast<std::array<uint16_t, 2>>(f);
    switch (std::fpclassify(f)) {
      case FP_SUBNORMAL:
      case FP_ZERO:
        // sign preserving zero (denormal go to zero)
        raw_bits_ = iraw[1];
        raw_bits_ &= 0x8000;
        break;
      case FP_INFINITE:
        raw_bits_ = iraw[1];
        break;
      case FP_NAN:
        // truncate and set MSB of the mantissa force QNAN
        raw_bits_ = iraw[1];
        raw_bits_ |= 1 << 6;
        break;
      case FP_NORMAL:
        // round to nearest even and truncate
        const uint32_t rounding_bias = 0x00007FFF + (iraw[1] & 0x1);
        const uint32_t int_raw = bit_cast<uint32_t>(f) + rounding_bias;
        iraw = bit_cast<std::array<uint16_t, 2>>(int_raw);
        raw_bits_ = iraw[1];
        break;
    }
    return *this;
  }

  template <typename IntegerType,
            typename SFINAE = typename std::enable_if<
                std::is_integral<IntegerType>::value>::type>
  bfloat16_t &operator=(const IntegerType i) {
    // Call the converting constructor that is optimized for integer types,
    // followed by the fast defaulted move-assignment operator.
    return (*this) = bfloat16_t{i};
  }

  operator float() const {
    std::array<uint16_t, 2> iraw = {{0, raw_bits_}};
    return bit_cast<float>(*(uint32_t*)(iraw.data()));
  }

  bfloat16_t &operator+=(const float a) {
    (*this) = float{*this} + a;
    return *this;
  }
  bfloat16_t &operator*=(const float a) {
    (*this) = float{*this} * a;
    return *this;
  }
  bfloat16_t &operator-=(const float a) {
    (*this) = float{*this} - a;
    return *this;
  }
  bfloat16_t &operator/=(const float a) {
    (*this) = float{*this} / a;
    return *this;
  }

 private:
  // Converts the 32 bits of a normal float or zero to the bits of a bfloat16.
  static constexpr uint16_t convert_bits_of_normal_or_zero(
      const uint32_t bits) {
    return static_cast<uint16_t>(
        uint32_t{bits + uint32_t{0x7FFFU + (uint32_t{bits >> 16} & 1U)}} >> 16);
  }
};

static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must be 2 bytes");

void cvt_float_to_bfloat16(bfloat16_t *out, const float *inp, size_t nelems);
void cvt_bfloat16_to_float(float *out, const bfloat16_t *inp, size_t nelems);

// performs element-by-element sum of inp and add float arrays and stores
// result to bfloat16 out array with downconversion
// out[:] = (bfloat16_t)(inp0[:] + inp1[:])
void add_floats_and_cvt_to_bfloat16(bfloat16_t *out, const float *inp0,
                                    const float *inp1, size_t nelems);

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

}  // namespace xir
