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

#include "cpu_std_inc.hpp"
#include "cpu_util.hpp"

namespace vart {
namespace cpu {

struct FMapAttr;
using fmap_coord_t = FMapAttr;

typedef struct FMapAttr {
  std::int64_t n;
  std::int64_t h;
  std::int64_t w;
  std::int64_t d;
  std::int64_t c;

  FMapAttr() : n(0), h(0), w(0), d(1), c(0) {}
  FMapAttr(std::int64_t var_n, std::int64_t var_h, std::int64_t var_w,
           std::int64_t var_c)
      : n(var_n), h(var_h), w(var_w), d(1), c(var_c) {}
  FMapAttr(std::int64_t var_n, std::int64_t var_h, std::int64_t var_w,
           std::int64_t var_d, std::int64_t var_c)
      : n(var_n), h(var_h), w(var_w), d(var_d), c(var_c) {}
  FMapAttr(const vector<int>& dims) {
    auto DIM_SIZE = dims.size();
    if (DIM_SIZE < 1 || DIM_SIZE > 5) {
      UNI_LOG_ERROR(VART_NOT_SUPPORT)
          << "Not support dim: " << Vec2Str(dims, ", ") << endl;
      abort();
    }
    n = h = w = d = c = 1;
    vector<int> dims_;
    dims_.insert(dims_.begin(), dims.begin(), dims.end());
    while (dims_.size() < 4) {
      dims_.insert(dims_.begin(), 1);
    }
    n = dims_[0];
    h = dims_[1];
    w = dims_[2];
    if (dims_.size() == 4) {
      c = dims_[3];
    } else if (dims_.size() == 5) {
      d = dims_[3];
      c = dims_[4];
    }
  }

  std::int64_t num() const { return n * h * w * d * c; }
  std::int64_t ncod() const { return h * w * d * c; }
  std::int64_t hcod() const { return w * d * c; }
  std::int64_t wcod() const { return d * c; }
  std::int64_t dcod() const { return c; }
  std::int64_t ccod() const { return 1; }

  inline std::int64_t coord2pos(const fmap_coord_t& coord) const {
    int64_t pos = 0;

    pos += coord.n * ncod();
    pos += coord.h * hcod();
    pos += coord.w * wcod();
    pos += coord.d * dcod();
    pos += coord.c * ccod();

    return pos;
  }
  inline fmap_coord_t pos2coord(std::int64_t pos) const {
    fmap_coord_t coord;

    coord.n = pos / ncod();
    coord.h = (pos % ncod()) / hcod();
    coord.w = ((pos % ncod()) % hcod()) / wcod();
    coord.d = (((pos % ncod()) % hcod()) % wcod()) / dcod();
    coord.c = (((pos % ncod()) % hcod()) % wcod() % dcod()) / ccod();

    return coord;
  }

  void print_param(const string& head = "fmap") const {
    UNI_LOG_DEBUG_INFO << head << ": ["
                       << "n: " << n << ", h: " << h << ", w: " << w
                       << ", d: " << d << ", c: " << c << "]" << endl;
  }

  bool operator==(const FMapAttr& other) {
    if (n == other.n && h == other.h && w == other.w && d == other.d &&
        c == other.c) {
      return true;
    }

    return false;
  }
} FMap_t;

typedef struct DilationAttr {
  int h;
  int w;
  int d;

  DilationAttr() : h(1), w(1), d(1) {}
  DilationAttr(int var) : h(var), w(var), d(var) {}
  DilationAttr(int var_h, int var_w) : h(var_h), w(var_w), d(1) {}
  DilationAttr(int var_h, int var_w, int var_d)
      : h(var_h), w(var_w), d(var_d) {}

  void print_param(const string& head = "stride") const {
    UNI_LOG_DEBUG_INFO << head << ": ["
                       << "h: " << h << ", w: " << w << " ,d: " << d << "]"
                       << endl;
  }
  bool operator==(const DilationAttr& other) {
    if (h == other.h && w == other.w && d == other.d) {
      return true;
    }

    return false;
  }

} Dilation_t;

typedef struct KernelAttr {
  int ori_h;
  int ori_w;
  int ori_d;
  int dilation_h;
  int dilation_w;
  int dilation_d;
  int h;
  int w;
  int d;

  KernelAttr()
      : ori_h(1),
        ori_w(1),
        ori_d(1),
        dilation_h(1),
        dilation_w(1),
        dilation_d(1),
        h(1),
        w(1),
        d(1) {}
  KernelAttr(int var_h, int var_w)
      : ori_h(var_h),
        ori_w(var_w),
        ori_d(1),
        dilation_h(1),
        dilation_w(1),
        dilation_d(1),
        h(var_h),
        w(var_w),
        d(1) {}
  KernelAttr(int var_h, int var_w, int var_d)
      : ori_h(var_h),
        ori_w(var_w),
        ori_d(var_d),
        dilation_h(1),
        dilation_w(1),
        dilation_d(1),
        h(var_h),
        w(var_w),
        d(var_d) {}
  KernelAttr(int var_h, int var_w, Dilation_t var_dilation)
      : ori_h(var_h),
        ori_w(var_w),
        ori_d(1),
        dilation_h(var_dilation.h),
        dilation_w(var_dilation.w),
        dilation_d(1),
        d(1) {
    h = ori_h * dilation_h - (dilation_h - 1);
    w = ori_w * dilation_w - (dilation_w - 1);
  }
  KernelAttr(int var_h, int var_w, int var_d, Dilation_t var_dilation)
      : ori_h(var_h),
        ori_w(var_w),
        ori_d(var_d),
        dilation_h(var_dilation.h),
        dilation_w(var_dilation.w),
        dilation_d(var_dilation.d) {
    h = ori_h * dilation_h - (dilation_h - 1);
    w = ori_w * dilation_w - (dilation_w - 1);
    d = ori_d * dilation_d - (dilation_d - 1);
  }

  void print_param(const string& head = "kernel") const {
    UNI_LOG_DEBUG_INFO << head << ": ["
                       << "ori_h: " << ori_h << ", ori_w: " << ori_w
                       << ", ori_d: " << ori_d << ", dilation_h: " << dilation_h
                       << ", dilation_w: " << dilation_w
                       << ", dilation_d:  " << dilation_d << ", h: " << h
                       << ", w: " << w << ", d: " << d << "]" << endl;
  }

  bool operator==(const KernelAttr& other) {
    if (ori_h == other.ori_h && ori_w == other.ori_w && ori_d == other.ori_d &&
        dilation_h == other.dilation_h && dilation_w == other.dilation_w &&
        dilation_d == other.dilation_d && h == other.h && w == other.w &&
        d == other.d) {
      return true;
    }

    return false;
  }

} Kernel_t;

typedef struct StrideAttr {
  int h;
  int w;
  int d;

  StrideAttr() : h(1), w(1), d(1) {}
  StrideAttr(int var_h, int var_w) : h(var_h), w(var_w), d(1) {}
  StrideAttr(int var_h, int var_w, int var_d) : h(var_h), w(var_w), d(var_d) {}

  void print_param(const string& head = "stride") const {
    UNI_LOG_DEBUG_INFO << head << ": ["
                       << "h: " << h << ", w: " << w << " ,d: " << d << "]"
                       << endl;
  }
  bool operator==(const StrideAttr& other) {
    if (h == other.h && w == other.w && d == other.d) {
      return true;
    }

    return false;
  }

} Stride_t;

typedef struct ScaleAttr {
  int h;
  int w;
  int d;

  ScaleAttr() : h(1), w(0), d(1) {}
  ScaleAttr(int var_h, int var_w) : h(var_h), w(var_w), d(1) {}
  ScaleAttr(int var_h, int var_w, int var_d) : h(var_h), w(var_w), d(var_d) {}

  void print_param(const string& head = "scale") const {
    UNI_LOG_DEBUG_INFO << head << ": ["
                       << "h: " << h << ", w: " << w << ", d: " << d << "]"
                       << endl;
  }
  void check_same(const ScaleAttr& other) const {
    UNI_LOG_CHECK(h == other.h, VART_INVALID_VALUE)
        << ", " << h << " != " << other.h;
    UNI_LOG_CHECK(w == other.w, VART_INVALID_VALUE)
        << ", " << w << " != " << other.w;
    UNI_LOG_CHECK(d == other.d, VART_INVALID_VALUE)
        << ", " << d << " != " << other.d;
  }
} Scale_t;

typedef struct PadAttr {
  enum {
    PAD_MODE_FLOOR,
    PAD_MODE_CEIL,
    PAD_MODE_SAME,   // make output with same width and height as input
    PAD_MODE_VALID,  // no padding
    PAD_MODE_CONSTANT,
    PAD_MODE_REFLECT,
    PAD_MODE_SYMMETRIC,
  };
  int mode;
  int l;  // left
  int t;  // top
  int n;  // near
  int r;  // right
  int b;  // bottom
  int f;  // far

  PadAttr() : mode(PAD_MODE_FLOOR), l(0), t(0), n(0), r(0), b(0), f(0) {}
  PadAttr(int var_l, int var_t, int var_r, int var_b)
      : mode(PAD_MODE_FLOOR),
        l(var_l),
        t(var_t),
        n(0),
        r(var_r),
        b(var_b),
        f(0) {}
  PadAttr(int var_l, int var_t, int var_n, int var_r, int var_b, int var_f)
      : mode(PAD_MODE_FLOOR),
        l(var_l),
        t(var_t),
        n(var_n),
        r(var_r),
        b(var_b),
        f(var_f) {}
  PadAttr(int var_mode, int var_l, int var_t, int var_r, int var_b)
      : mode(var_mode), l(var_l), t(var_t), n(0), r(var_r), b(var_b), f(0) {}
  PadAttr(int var_mode, int var_l, int var_t, int var_n, int var_r, int var_b,
          int var_f)
      : mode(var_mode),
        l(var_l),
        t(var_t),
        n(var_n),
        r(var_r),
        b(var_b),
        f(var_f) {}
  PadAttr(std::string var_mode, int var_l, int var_t, int var_r, int var_b)
      : l(var_l), t(var_t), n(0), r(var_r), b(var_b), f(0) {
    if (var_mode == "FLOOR")
      mode = PAD_MODE_FLOOR;
    else if (var_mode == "CEIL")
      mode = PAD_MODE_CEIL;
    else if (var_mode == "SAME")
      mode = PAD_MODE_SAME;
    else if (var_mode == "VALID")
      mode = PAD_MODE_VALID;
    else if (var_mode == "CONSTANT")
      mode = PAD_MODE_CONSTANT;
    else if (var_mode == "REFLECT")
      mode = PAD_MODE_REFLECT;
    else if (var_mode == "SYMMETRIC")
      mode = PAD_MODE_SYMMETRIC;
    else
      UNI_LOG_FATAL(VART_NOT_SUPPORT)
          << "We only support pad_mode: FLOOR, CEIL, SAME, VALID. "
             "For Pad op, we support pad_mode: CONSTANT, REFLECT, SYMMETRIC"
          << endl;
  }
  PadAttr(std::string var_mode, int var_l, int var_t, int var_n, int var_r,
          int var_b, int var_f)
      : l(var_l), t(var_t), n(var_n), r(var_r), b(var_b), f(var_f) {
    if (var_mode == "FLOOR")
      mode = PAD_MODE_FLOOR;
    else if (var_mode == "CEIL")
      mode = PAD_MODE_CEIL;
    else if (var_mode == "SAME")
      mode = PAD_MODE_SAME;
    else if (var_mode == "VALID")
      mode = PAD_MODE_VALID;
    else if (var_mode == "CONSTANT")
      mode = PAD_MODE_CONSTANT;
    else if (var_mode == "REFLECT")
      mode = PAD_MODE_REFLECT;
    else if (var_mode == "SYMMETRIC")
      mode = PAD_MODE_SYMMETRIC;
    else
      UNI_LOG_FATAL(VART_NOT_SUPPORT)
          << "We only support pad_mode: FLOOR, CEIL, SAME, VALID. "
             "For Pad op, we support pad_mode: CONSTANT, REFLECT, SYMMETRIC"
          << endl;
  }

  void print_param(const string& head = "pad") const {
    UNI_LOG_DEBUG_INFO << head << ": ["
                       << "mode: " << mode << ", left: " << l << ", top: " << t
                       << ", near: " << n << ", right: " << r
                       << ", bottom: " << b << ", far: " << f << "]" << endl;
  }

  bool operator==(const PadAttr& other) {
    if (mode == other.mode && l == other.l && t == other.t && r == other.r &&
        b == other.b && f == other.f && n == other.n) {
      return true;
    }

    return false;
  }
} Pad_t;

}  // namespace cpu
}  // namespace vart
