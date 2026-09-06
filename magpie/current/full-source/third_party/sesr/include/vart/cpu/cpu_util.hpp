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

#include <filesystem>
#include <limits>

#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(ORT_ROUNDING_MODE, "0")
DEF_ENV_PARAM(ORT_ROUNDING_MODE_EVEN, "0")
#include "cpu_std_inc.hpp"

namespace vart {
namespace cpu {

enum DataFmt {
  DATA_FMT_MIN = 0,
  DATA_FMT_BIN = DATA_FMT_MIN,
  DATA_FMT_DEC,                         // 1
  DATA_FMT_HEX_CONT_SMALLEND,           // 2
  DATA_FMT_HEX_CONT_BIGEND,             // 3
  DATA_FMT_HEX_CONT_SMALLEND_BANKADDR,  // 4
  DATA_FMT_HEX_CONT_BIGEND_BANKADDR,    // 5
  DATA_FMT_HEX_CONT_SMALLEND_DDRADDR,   // 6
  DATA_FMT_HEX_CONT_BIGEND_DDRADDR,     // 7
  DATA_FMT_MAX,
};

enum SaveMode {
  SM_MIN = 0,
  SM_TRUNC = SM_MIN,
  SM_APPEND,
  SM_MAX,
};

namespace fs = std::filesystem;
// log folder and file operation
template <typename T>
void ChkOpen(T& f, const string& fname) {
  auto err_code = error_code();
  auto stat = fs::status(fname, err_code);
  if (!fs::exists(stat)) {
    UNI_LOG_ERROR(VART_FILE_ERROR)
        << fname << " stat error: " << err_code.message() << endl;
  }

  // handle common file
  if (!fs::is_regular_file(stat)) {
    UNI_LOG_ERROR(VART_FILE_ERROR) << "Not a normal file: " << fname << endl;
  }

  if (f.fail()) {
    UNI_LOG_ERROR(VART_FILE_ERROR) << "open file fail(): " << fname << endl;
    abort();
  }

  if (f.bad()) {
    UNI_LOG_ERROR(VART_FILE_ERROR) << "open file bad(): " << fname << endl;
    abort();
  }

  if (f.eof()) {
    UNI_LOG_ERROR(VART_FILE_ERROR) << "open file eof(): " << fname << endl;
    abort();
  }
}

void DelFileOrDir(const string& name);
bool ChkFile(const string& name, bool del_exists = false);
bool ChkFolder(const string& name, bool del_exists = false);
uint64_t GetFileSize(const string& fname);
uint64_t GetFileLine(const string& fname);

// load funcs: only SaveDec using templates
template <typename T>
uint64_t LoadDec(const string& load_name, T* data, uint64_t size) {
  string word;
  std::fstream f(load_name);
  vector<string> ori_data;

  ChkOpen(f, load_name);

  uint64_t num = 0;
  while (f >> word) {
    if (num > size) {
      UNI_LOG_DEBUG_WARNING << "file size exceeds memory size: " << load_name;
      break;
    }

    try {
      data[num++] = stoll(word, nullptr, 10);
    } catch (std::invalid_argument& e) {
      UNI_LOG_ERROR(VART_INVALIDE_PARAM) << e.what() << " invalid argument!";
      abort();
    }
  }

  return num;
}
uint64_t LoadBin(const string& load_name, char* data, uint64_t size);

// save funcs: only SaveDec using templates
template <typename T>
void SaveDec(const string& save_name, const T* data, uint64_t size,
             int line_size, int mode) {
  // write into file
  std::ofstream f;

  if (mode == SM_TRUNC) {
    f.open(save_name, std::ios::trunc);
  } else {
    f.open(save_name, std::ios::app);
  }

  ChkOpen(f, save_name);

  // line related vars
  int line_num = size / line_size + 1;
  int line_left = size % line_size;

  // write line_num lines
  for (int i = 0; i < line_num; i++) {
    // handle last line, it's special
    if (i == line_num - 1 && line_left == 0) break;

    // write line_size data every line
    for (int j = 0; j < line_size; j++) {
      // handle last line, it's special
      if (i == line_num - 1 && j >= line_left) break;

      // every 4 data elements, insert 4 spaces
      if (j != 0 && j % 4 == 0)
        f << " "
          << " "
          << " "
          << " ";

      // save data using related format
      if (line_size == 1) {
        f << dec << +data[i * line_size + j];
      } else {
        if (sizeof(T) == 1) {
          f << dec << setw(6) << setfill(' ') << +data[i * line_size + j];
        } else if (sizeof(T) == 2) {
          f << dec << setw(9) << setfill(' ') << +data[i * line_size + j];
        } else if (sizeof(T) == 4) {
          f << dec << setw(12) << setfill(' ') << +data[i * line_size + j];
        } else if (sizeof(T) == 8) {
          f << dec << setw(24) << setfill(' ') << +data[i * line_size + j];
        } else {
          UNI_LOG_ERROR(VART_NOT_SUPPORT)
              << "Not support type, sizeof(T) = " << sizeof(T) << endl;
          abort();
        }
      }
    }
    f << endl;
  }

  f.close();
}

void SaveBin(const string& save_name, const char* data, uint64_t size,
             int mode = SM_TRUNC);

// misc
template <typename T>
T Random(T low, T high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dis(low, high);

  return dis(gen);
}

template <typename T>
void Random(T* data, uint64_t size, T low, T high, uint64_t seed) {
  std::random_device rd;
  std::mt19937 gen((int)seed);
  std::uniform_int_distribution<short> dis(low, high);

  for (auto i = 0U; i < size; i++) {
    if (seed == 0)
      data[i] = 0;
    else
      data[i] = (T)dis(gen);
  }
}
template void Random<char>(char* data, uint64_t size, char low, char high,
                           uint64_t seed);

bool Str2Bool(const string& str);
string Bool2Str(bool flag);

template <typename T>
string Vec2Str(const vector<T>& v, const string& split) {
  string str;

  for (auto i = 0U; i < v.size(); i++) {
    str += to_string(v[i]);
    if (i != v.size() - 1) str += split;
  }

  return str;
}

int64_t Str2S64(const string& str);
uint64_t Str2U64(const string& str);

uint32_t FillMaskU32(int start, int num);
uint64_t FillMaskU64(int start, int num);

uint64_t AlignByN(uint64_t data, uint64_t align);

string GetFileNameSuffix(int fmt);

// handle string
string Trim(const string& str);
vector<string> Split(const string& str, const string& delim = " ,:;\t",
                     bool trim_flag = true);
string SplitFirst(const string& str, const string& delim = " ,:;\t",
                  bool trim_flag = true);

// used to calculate time
enum TimeType {
  TIME_UNIT_S,
  TIME_UNIT_MS,
  TIME_UNIT_US,
};

typedef std::chrono::time_point<chrono::steady_clock> time_stamp_t;
time_stamp_t time_start();
unsigned long time_finish(const time_stamp_t& start, int type = TIME_UNIT_MS);

template <typename T>
T round_to_even(double data, T data_min = std::numeric_limits<T>::lowest(),
                T data_max = std::numeric_limits<T>::max()) {
  T rlt = 0;
  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else if ((data - floor(data)) == 0.5) {
    rlt = std::round(data * 0.5f) * 2.0f;
  } else {
    rlt = static_cast<T>(round(data));
  }
  return rlt;
}

float DPURound(double data, float data_min, float data_max);
template <typename T>
T DPURound(double data, T data_min = std::numeric_limits<T>::lowest(),
           T data_max = std::numeric_limits<T>::max()) {
  T rlt = 0;

  if (ENV_PARAM(ORT_ROUNDING_MODE)) {
    if (data > data_max) {
      rlt = data_max;
    } else if (data < data_min) {
      rlt = data_min;
    } else if ((data - floor(data)) == 0.5) {
      rlt = std::round(data * 0.5f) * 2.0f;
    } else {
      rlt = static_cast<T>(round(data));
    }
  } else if (ENV_PARAM(ORT_ROUNDING_MODE_EVEN)) {
    if (data > data_max) {
      rlt = data_max;
    } else if (data < data_min) {
      rlt = data_min;
    } else {
      if (data - floor(data) == 0.5) {
        if ((int)floor(data) % 2 == 0)
          data = floor(data);
        else
          data = ceil(data);
      }
      rlt = static_cast<T>(round(data));
    }
  } else {
    if (data > data_max) {
      rlt = data_max;
    } else if (data < data_min) {
      rlt = data_min;
    } else if (data < 0 && (data - floor(data)) == 0.5) {
      rlt = static_cast<T>(ceil(data));
    } else {
      rlt = static_cast<T>(round(data));
    }
  }

  return rlt;
}

float STDRound(double data, float data_min, float data_max);
template <typename T>
T STDRound(double data, T data_min = std::numeric_limits<T>::lowest(),
           T data_max = std::numeric_limits<T>::max()) {
  T rlt = 0;
  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else {
    double floor_value = std::floor(data);
    double ceiling_value = std::ceil(data);
    double diff_floor = data - floor_value;
    double diff_ceiling = ceiling_value - data;
    if (diff_floor == diff_ceiling) {
      // Round to the nearest even
      if (std::fmod(floor_value, 2.0) == 0.0) {
        rlt = static_cast<T>(floor_value);
      } else {
        rlt = static_cast<T>(ceiling_value);
      }
    } else {
      rlt = (diff_floor < diff_ceiling) ? floor_value : ceiling_value;
      rlt = static_cast<T>(rlt);
    }
  }
  return rlt;
}

float Py3Round(float input);

template <typename T>
T Py3Round(double data, T data_min = std::numeric_limits<T>::lowest(),
           T data_max = std::numeric_limits<T>::max()) {
  T rlt = 0;
  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else {
    auto default_round_mode = std::fegetround();
    if (!(FE_TONEAREST == default_round_mode)) {
      std::fesetround(FE_TONEAREST);
      rlt = std::nearbyint(data);
      std::fesetround(default_round_mode);
    } else {
      rlt = std::nearbyint(data);
    }
  }
  return rlt;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
round_normal(const std::string& round_mode_, double data,
               T data_min = std::numeric_limits<T>::lowest(),
               T data_max = std::numeric_limits<T>::max()) {
  T rlt = T();
  if (round_mode_ == "STD_ROUND") {
    rlt = STDRound(data, data_min, data_max);
  } else if (round_mode_ == "DPU_ROUND") {
    rlt = DPURound(data, data_min, data_max);
  } else if (round_mode_ == "PY3_ROUND") {
    rlt = Py3Round(data, data_min, data_max);
  } else {
    UNI_LOG_ERROR(VART_NOT_SUPPORT)
        << "Not supported round mode " << round_mode_ << endl;
    abort();
  }
  return rlt;
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
round_normal(const std::string& round_mode_, double data,
               T data_min = std::numeric_limits<T>::lowest(),
               T data_max = std::numeric_limits<T>::max()) {
  T rlt = T(data);
  return rlt;
}

template <typename T>
void NHWC2NCHW(int n, int h, int w, int c, const T* src, T* dst) {
  for (int idx_n = 0; idx_n < n; idx_n++)
    for (int idx_c = 0; idx_c < c; idx_c++) {
      for (int idx_h = 0; idx_h < h; idx_h++) {
        for (int idx_w = 0; idx_w < w; idx_w++) {
          dst[idx_n * h * w * c + idx_c * h * w + idx_h * w + idx_w] =
              src[idx_n * h * w * c + idx_h * w * c + idx_w * c + idx_c];
        }
      }
    }
}

template <typename T>
void NCHW2NHWC(int n, int c, int h, int w, const T* src, T* dst) {
  for (int idx_n = 0; idx_n < n; idx_n++)
    for (int idx_h = 0; idx_h < h; idx_h++)
      for (int idx_w = 0; idx_w < w; idx_w++)
        for (int idx_c = 0; idx_c < c; idx_c++)
          dst[idx_n * h * w * c + idx_h * w * c + idx_w * c + idx_c] =
              src[idx_n * h * w * c + idx_c * w * h + idx_h * w + idx_w];
}

template <typename T>
void HWC2CHW(int h, int w, int c, const T* src, T* dst) {
  for (int idx_c = 0; idx_c < c; idx_c++) {
    for (int idx_h = 0; idx_h < h; idx_h++) {
      for (int idx_w = 0; idx_w < w; idx_w++) {
        dst[idx_c * h * w + idx_h * w + idx_w] =
            src[idx_h * w * c + idx_w * c + idx_c];
      }
    }
  }
}

template <typename T>
void HWC2WCH(int h, int w, int c, const T* src, T* dst) {
  for (int idx_w = 0; idx_w < w; idx_w++) {
    for (int idx_c = 0; idx_c < c; idx_c++) {
      for (int idx_h = 0; idx_h < h; idx_h++) {
        dst[idx_w * c * h + idx_c * h + idx_h] =
            src[idx_h * w * c + idx_w * c + idx_c];
      }
    }
  }
}

template <typename T>
void CHW2HWC(int c, int h, int w, const T* src, T* dst) {
  for (int idx_h = 0; idx_h < h; idx_h++) {
    for (int idx_w = 0; idx_w < w; idx_w++) {
      for (int idx_c = 0; idx_c < c; idx_c++) {
        dst[idx_h * w * c + idx_w * c + idx_c] =
            src[idx_c * h * w + idx_h * w + idx_w];
      }
    }
  }
}

template <typename T>
void HW2WH(int h, int w, const T* src, T* dst) {
  for (int idx_w = 0; idx_w < w; idx_w++) {
    for (int idx_h = 0; idx_h < h; idx_h++) {
      dst[idx_w * h + idx_h] = src[idx_h * w + idx_w];
    }
  }
}

template <typename T>
void HWN2WHN(int h, int w, int n, const T* src, T* dst) {
  for (int idx_w = 0; idx_w < w; idx_w++) {
    for (int idx_h = 0; idx_h < h; idx_h++) {
      for (int idx_n = 0; idx_n < n; idx_n++) {
        dst[idx_w * h * n + idx_h * n + idx_n] =
            src[idx_h * w * n + idx_w * n + idx_n];
      }
    }
  }
}

template <typename T>
void CHWN2HWCN(int c, int h, int w, int n, const T* src, T* dst) {
  for (int idx_h = 0; idx_h < h; idx_h++) {
    for (int idx_w = 0; idx_w < w; idx_w++) {
      for (int idx_c = 0; idx_c < c; idx_c++) {
        for (int idx_n = 0; idx_n < n; idx_n++) {
          dst[idx_h * w * c * n + idx_w * c * n + idx_c * n + idx_n] =
              src[idx_c * h * w * n + idx_h * w * n + idx_w * n + idx_n];
        }
      }
    }
  }
}

/*
 * R = a*B + c
 * a : 1 x rows
 * B : rows x cols
 * c : 1 x cols
 * R : 1 x cols
 */
template <typename T>
void vecxmat_add_vec_rowaccess_c(int rows, int cols, const T* matA,
                                 const T* matB, const T* matC, T* matR) {
  for (int j = 0; j < cols; j++) {
    T rlt = 0;
    for (int i = 0; i < rows; i++) {
      rlt += matA[i] * matB[i * cols + j];
    }

    if (matC != nullptr) rlt += matC[j];
    matR[j] = rlt;
  }
}

template <typename T>
void softmax(T* src, int num, T* rlt) {
  T sum = 0;
  T max = *(std::max_element(src, src + num));
  for (auto i = 0; i < num; i++) {
    rlt[i] = exp(src[i] - max);
    sum += rlt[i];
  }
  for (auto i = 0; i < num; i++) {
    rlt[i] = rlt[i] / sum;
  }
}

template <typename T,
          typename R = typename std::conditional<
              std::is_floating_point<T>::value, T,
              typename std::conditional<std::is_signed<T>::value, int64_t,
                                        uint64_t>::type>::type>
R get_vec_mul(const vector<T>& v, uint32_t start_pos = 0) {
  if (v.size() == 0 || start_pos >= v.size()) return static_cast<R>(0);

  R rlt = static_cast<R>(1);
  for (auto i = start_pos; i < v.size(); i++) {
    rlt *= v[i];
  }

  return rlt;
}

template <>
bool get_vec_mul<bool>(const vector<bool>& v, uint32_t start_pos) = delete;

string tensor_name_2_save_name(string tensor_name);

template <typename SType, typename DType>
void dta_2_dtb(SType* src, DType* dst, int num) {
  if (std::is_same<SType, DType>::value) {
    std::copy_n(src, num, dst);
    return;
  }

  for (auto i = 0; i < num; i++) {
    dst[i] = static_cast<DType>(src[i]);
  }
}

std::int32_t get_shift_from_int32_rep(std::int32_t rep);
std::int32_t get_int32_representation_from_float(float value);
std::int32_t convert_float_to_qint(float in_f);
typedef union value_converter {
  std::uint32_t u;
  float f;
} value_converter_t;

std::uint32_t float_to_uint32(float data);

float uint32_to_float(std::uint32_t data);

float float_to_bf16(float data);
}  // namespace cpu
}  // namespace vart
