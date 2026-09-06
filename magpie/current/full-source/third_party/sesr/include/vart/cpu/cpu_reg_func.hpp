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
#include "cpu_tensor_buffer.hpp"

#define VART_STR_CONCAT_(__x, __y) cpu_runer_reg_func_handle__##__x##__y##_hook

#define VART_STR_CONCAT(__x, __y) VART_STR_CONCAT_(__x, __y)

#define REG_OP_INSTANCE_FUNC(xir_op_type, class_name)                          \
  extern "C" {                                                                 \
  vart::cpu::CPURegFunc& VART_STR_CONCAT(class_name, __COUNTER__) =            \
      vart::cpu::CPURegFunc::instance()->register_func(                        \
          std::string{xir_op_type},                                            \
          [](const xir::Subgraph* subg, const xir::Op* xir_op,                 \
             IMapTBs_t inputs, CPUTBPtr_t output) {                            \
            auto* tensor = xir_op->get_output_tensor();                        \
            auto xir_dtype = tensor->get_data_type().type;                     \
            auto bit_width = tensor->get_data_type().bit_width;                \
            string dbg_str = get_data_type_str(tensor);                        \
            vart::cpu::CPUOPBase* op = nullptr;                                \
            if (xir_dtype == xir::DataType::FLOAT) {                           \
              if (bit_width == 32) {                                           \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<float>>(                \
                             subg, xir_op, inputs, output);                    \
              } else if (bit_width == 64) {                                    \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<double>>(               \
                             subg, xir_op, inputs, output);                    \
              } else if (bit_width == 16) {                                    \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<int32_t>>(              \
                             subg, xir_op, inputs, output);                    \
              } else {                                                         \
                UNI_LOG_FATAL(VART_NOT_SUPPORT)                                \
                    << "Not support bit_width " << bit_width << endl;          \
              }                                                                \
            } else if (xir_dtype == xir::DataType::BFLOAT) {                   \
              if (bit_width == 16) {                                           \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<float>>(                \
                             subg, xir_op, inputs, output);                    \
              } else {                                                         \
                UNI_LOG_FATAL(VART_NOT_SUPPORT)                                \
                    << "Not support bit_width " << bit_width << endl;          \
              }                                                                \
            } else {                                                           \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<int32_t>>(                \
                           subg, xir_op, inputs, output);                      \
            }                                                                  \
            if (VART_DEBUG) {                                                  \
              UNI_LOG_DEBUG_INFO << "Construct vart::cpu::" << #class_name     \
                                 << "<" << dbg_str << "> obj" << endl;         \
            }                                                                  \
            return op;                                                         \
          });                                                                  \
  }

#define REG_BINARY_OP_INSTANCE_FUNC(xir_op_type, class_name)                   \
  extern "C" {                                                                 \
  vart::cpu::CPURegFunc& VART_STR_CONCAT(class_name, __COUNTER__) =            \
      vart::cpu::CPURegFunc::instance()->register_func(                        \
          std::string{xir_op_type},                                            \
          [](const xir::Subgraph* subg, const xir::Op* xir_op,                 \
             IMapTBs_t inputs, CPUTBPtr_t output) {                            \
            auto in0 =                                                         \
                xir_op->get_input_tensors("input")[0] -> get_data_type();      \
            auto in1 =                                                         \
                xir_op->get_input_tensors("input")[1] -> get_data_type();      \
            auto out = xir_op->get_output_tensor() -> get_data_type();         \
            vart::cpu::CPUOPBase* op = nullptr;                                \
            if (in0.type == xir::DataType::FLOAT && in0.bit_width == 32 &&     \
                in1.type == xir::DataType::FLOAT && in1.bit_width == 32 &&     \
                out.type == xir::DataType::FLOAT && out.bit_width == 32) {     \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<float, float, float>>(    \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 64 &&                                  \
                       in1.type == xir::DataType::FLOAT &&                     \
                       in1.bit_width == 32 &&                                  \
                       out.type == xir::DataType::FLOAT &&                     \
                       out.bit_width == 64) {                                  \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<double, float, double>>(  \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 32 &&                                  \
                       in1.type == xir::DataType::FLOAT &&                     \
                       in1.bit_width == 64 &&                                  \
                       out.type == xir::DataType::FLOAT &&                     \
                       out.bit_width == 64) {                                  \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<float, double, double>>(  \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 64 &&                                  \
                       in1.type == xir::DataType::FLOAT &&                     \
                       in1.bit_width == 64 &&                                  \
                       out.type == xir::DataType::FLOAT &&                     \
                       out.bit_width == 64) {                                  \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<double, double, double>>( \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::BFLOAT &&                    \
                       in0.bit_width == 16 &&                                  \
                       in1.type == xir::DataType::BFLOAT &&                    \
                       in1.bit_width == 16 &&                                  \
                       out.type == xir::DataType::BFLOAT &&                    \
                       out.bit_width == 16) {                                  \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<float, float, float>>(    \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type != xir::DataType::FLOAT &&                     \
                       in1.type != xir::DataType::FLOAT &&                     \
                       out.type != xir::DataType::FLOAT &&                     \
                       in0.type != xir::DataType::BFLOAT &&                    \
                       in1.type != xir::DataType::BFLOAT &&                    \
                       out.type != xir::DataType::BFLOAT) {                    \
              op = CPUOPFactory::Instance()                                    \
                       .create<                                                \
                           vart::cpu::class_name<int32_t, int32_t, int32_t>>(  \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type != xir::DataType::FLOAT &&                     \
                       in0.type != xir::DataType::BFLOAT &&                    \
                       in1.type == xir::DataType::FLOAT &&                     \
                       in1.bit_width == 32 &&                                  \
                       out.type == xir::DataType::FLOAT &&                     \
                       out.bit_width == 32) {                                  \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<int32_t, float, float>>(  \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 32 &&                                  \
                       in1.type != xir::DataType::FLOAT &&                     \
                       in1.type != xir::DataType::BFLOAT &&                    \
                       out.type == xir::DataType::FLOAT &&                     \
                       out.bit_width == 32) {                                  \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<float, int32_t, float>>(  \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type != xir::DataType::FLOAT &&                     \
                       in0.type != xir::DataType::BFLOAT &&                    \
                       in1.type == xir::DataType::FLOAT &&                     \
                       in1.bit_width == 64 &&                                  \
                       out.type == xir::DataType::FLOAT &&                     \
                       out.bit_width == 64) {                                  \
              op =                                                             \
                  CPUOPFactory::Instance()                                     \
                      .create<vart::cpu::class_name<int32_t, double, double>>( \
                          subg, xir_op, inputs, output);                       \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 64 &&                                  \
                       in1.type != xir::DataType::FLOAT &&                     \
                       in1.type != xir::DataType::BFLOAT &&                    \
                       out.type == xir::DataType::FLOAT &&                     \
                       out.bit_width == 64) {                                  \
              op =                                                             \
                  CPUOPFactory::Instance()                                     \
                      .create<vart::cpu::class_name<double, int32_t, double>>( \
                          subg, xir_op, inputs, output);                       \
            } else {                                                           \
              UNI_LOG_FATAL(VART_NOT_SUPPORT)                                  \
                  << "Not support " << #class_name << "<" << in0.to_string()   \
                  << ", " << in1.to_string() << ", " << out.to_string() << ">" \
                  << endl;                                                     \
            }                                                                  \
            if (VART_DEBUG) {                                                  \
              UNI_LOG_DEBUG_INFO << "Construct vart::cpu::" << #class_name     \
                                 << "<" << in0.to_string() << ", "             \
                                 << in1.to_string() << ", " << out.to_string() \
                                 << "> obj" << endl;                           \
            }                                                                  \
            return op;                                                         \
          });                                                                  \
  }

#define REG_SPECIFIED_DATATYPE_OP_INSTANCE_FUNC(xir_op_type, class_name)       \
  extern "C" {                                                                 \
  vart::cpu::CPURegFunc& VART_STR_CONCAT(class_name, __COUNTER__) =            \
      vart::cpu::CPURegFunc::instance()->register_func(                        \
          std::string{xir_op_type},                                            \
          [](const xir::Subgraph* subg, const xir::Op* xir_op,                 \
             IMapTBs_t inputs, CPUTBPtr_t output) {                            \
            auto in0 =                                                         \
                xir_op->get_input_tensors("input")[0] -> get_data_type();      \
            auto out = xir_op->get_output_tensor() -> get_data_type();         \
            vart::cpu::CPUOPBase* op = nullptr;                                \
            if(in0.type == xir::DataType::UINT && in0.bit_width == 16          \
               && out.type == xir::DataType::FLOAT) {                          \
               op = CPUOPFactory::Instance()                                   \
                         .create<vart::cpu::class_name<uint16_t, float>>(      \
                             subg, xir_op, inputs, output);                    \
            } else if (in0.type == xir::DataType::FLOAT && in0.bit_width == 32) {     \
              if (out.type == xir::DataType::FLOAT && out.bit_width == 32) {   \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<float, float>>(         \
                             subg, xir_op, inputs, output);                    \
              } else if (out.type == xir::DataType::FLOAT &&                   \
                         out.bit_width == 64) {                                \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<float, double>>(        \
                             subg, xir_op, inputs, output);                    \
              } else if (out.type == xir::DataType::BFLOAT &&                  \
                         out.bit_width == 16) {                                \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<float, float>>(         \
                             subg, xir_op, inputs, output);                    \
              } else if (out.type == xir::DataType::INT &&                     \
                         out.bit_width == 64) {                                \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<float, int64_t>>(       \
                             subg, xir_op, inputs, output);                    \
              } else if (out.type == xir::DataType::INT &&                     \
                         out.bit_width == 32) {                                \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<float, int32_t>>(       \
                             subg, xir_op, inputs, output);                    \
              } else if ((out.type == xir::DataType::XINT ||                   \
                          out.type == xir::DataType::XUINT) &&                 \
                         (out.bit_width == 8 || out.bit_width == 16)) {        \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<float, int32_t>>(       \
                             subg, xir_op, inputs, output);                    \
              } else {                                                         \
                UNI_LOG_FATAL(VART_NOT_SUPPORT)                                \
                    << "Not support " << #class_name << "<" << in0.to_string() \
                    << ", " << out.to_string() << ">" << endl;                 \
              }                                                                \
            } else if ((in0.type == xir::DataType::INT ||                      \
                        in0.type == xir::DataType::XINT ||                     \
                        in0.type == xir::DataType::XUINT) &&                   \
                       in0.bit_width <= 32) {                                  \
              if ((out.type == xir::DataType::INT ||                           \
                   out.type == xir::DataType::XINT ||                          \
                   out.type == xir::DataType::XUINT) &&                        \
                  out.bit_width <= 32) {                                       \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<int32_t, int32_t>>(     \
                             subg, xir_op, inputs, output);                    \
              } else if (out.type == xir::DataType::FLOAT &&                   \
                         out.bit_width == 32) {                                \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<int32_t, float>>(       \
                             subg, xir_op, inputs, output);                    \
              } else if (out.type == xir::DataType::FLOAT &&                   \
                         out.bit_width == 64) {                                \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<int32_t, double>>(      \
                             subg, xir_op, inputs, output);                    \
              } else if (out.type == xir::DataType::INT &&                     \
                         out.bit_width == 64) {                                \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<int32_t, int64_t>>(     \
                             subg, xir_op, inputs, output);                    \
              } else if (out.type == xir::DataType::BFLOAT &&                  \
                         out.bit_width == 16) {                                \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<int32_t, float>>(       \
                             subg, xir_op, inputs, output);                    \
              } else {                                                         \
                UNI_LOG_FATAL(VART_NOT_SUPPORT)                                \
                    << "Not support " << #class_name << "<" << in0.to_string() \
                    << ", " << out.to_string() << ">" << endl;                 \
              }                                                                \
            } else if (in0.type == xir::DataType::BFLOAT &&                    \
                       in0.bit_width == 16) {                                  \
              if (out.type == xir::DataType::FLOAT && out.bit_width == 32) {   \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<float, float>>(         \
                             subg, xir_op, inputs, output);                    \
              } else if ((out.type == xir::DataType::INT ||                    \
                          out.type == xir::DataType::XINT ||                   \
                          out.type == xir::DataType::XUINT) &&                 \
                         out.bit_width == 32) {                                \
                op = CPUOPFactory::Instance()                                  \
                         .create<vart::cpu::class_name<float, int32_t>>(       \
                             subg, xir_op, inputs, output);                    \
              } else {                                                         \
                UNI_LOG_FATAL(VART_NOT_SUPPORT)                                \
                    << "Not support " << #class_name << "<" << in0.to_string() \
                    << ", " << out.to_string() << ">" << endl;                 \
              }                                                                \
            } else {                                                           \
              UNI_LOG_FATAL(VART_NOT_SUPPORT)                                  \
                  << "Not support " << #class_name << "<" << in0.to_string()   \
                  << ", " << out.to_string() << ">" << endl;                   \
            }                                                                  \
            if (VART_DEBUG) {                                                  \
              UNI_LOG_DEBUG_INFO << "Construct vart::cpu::" << #class_name     \
                                 << "<" << in0.to_string() << ", "             \
                                 << out.to_string() << "> obj" << endl;        \
            }                                                                  \
            return op;                                                         \
          });                                                                  \
  }

#define REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC(                       \
    xir_op_type, class_name, weight_name)                                      \
  extern "C" {                                                                 \
  vart::cpu::CPURegFunc& VART_STR_CONCAT(class_name, __COUNTER__) =            \
      vart::cpu::CPURegFunc::instance()->register_func(                        \
          std::string{xir_op_type},                                            \
          [](const xir::Subgraph* subg, const xir::Op* xir_op,                 \
             IMapTBs_t inputs, CPUTBPtr_t output) {                            \
            auto in0 =                                                         \
                xir_op->get_input_tensors("input")[0] -> get_data_type();      \
            auto wgt = xir_op->get_input_ops(weight_name)[0]                   \
                       -> get_output_tensor() -> get_data_type();              \
            vart::cpu::CPUOPBase* op = nullptr;                                \
            if (in0.type == xir::DataType::FLOAT && in0.bit_width == 32 &&     \
                wgt.type == xir::DataType::FLOAT && wgt.bit_width == 32) {     \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<float, float>>(           \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 32 &&                                  \
                       wgt.type == xir::DataType::FLOAT &&                     \
                       wgt.bit_width == 64) {                                  \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<float, double>>(          \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 32 &&                                  \
                       (wgt.type == xir::DataType::INT ||                      \
                        wgt.type == xir::DataType::XINT ||                     \
                        wgt.type == xir::DataType::XUINT) &&                   \
                       wgt.bit_width <= 32) {                                  \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<float, int32_t>>(         \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 64 &&                                  \
                       wgt.type == xir::DataType::FLOAT &&                     \
                       wgt.bit_width == 32) {                                  \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<double, float>>(          \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 64 &&                                  \
                       wgt.type == xir::DataType::FLOAT &&                     \
                       wgt.bit_width == 64) {                                  \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<double, double>>(         \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 64 &&                                  \
                       (wgt.type == xir::DataType::INT ||                      \
                        wgt.type == xir::DataType::XINT ||                     \
                        wgt.type == xir::DataType::XUINT) &&                   \
                       wgt.bit_width <= 32) {                                  \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<double, int32_t>>(        \
                           subg, xir_op, inputs, output);                      \
            } else if ((in0.type == xir::DataType::INT ||                      \
                        in0.type == xir::DataType::XINT ||                     \
                        in0.type == xir::DataType::XUINT) &&                   \
                       in0.bit_width <= 32 &&                                  \
                       wgt.type == xir::DataType::FLOAT &&                     \
                       wgt.bit_width == 32) {                                  \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<int32_t, float>>(         \
                           subg, xir_op, inputs, output);                      \
            } else if ((in0.type == xir::DataType::INT ||                      \
                        in0.type == xir::DataType::XINT ||                     \
                        in0.type == xir::DataType::XUINT) &&                   \
                       in0.bit_width <= 32 &&                                  \
                       wgt.type == xir::DataType::FLOAT &&                     \
                       wgt.bit_width == 64) {                                  \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<int32_t, double>>(        \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::BFLOAT &&                    \
                       in0.bit_width == 16 &&                                  \
                       wgt.type == xir::DataType::BFLOAT &&                    \
                       wgt.bit_width == 16) {                                  \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<float, float>>(           \
                           subg, xir_op, inputs, output);                      \
            } else {                                                           \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<int32_t, int32_t>>(       \
                           subg, xir_op, inputs, output);                      \
            }                                                                  \
            if (VART_DEBUG) {                                                  \
              UNI_LOG_DEBUG_INFO << "Construct vart::cpu::" << #class_name     \
                                 << "<" << in0.to_string() << ", "             \
                                 << wgt.to_string() << "> obj" << endl;        \
            }                                                                  \
            return op;                                                         \
          });                                                                  \
  }

#define REG_SPECIFIED_MATMUL_DATATYPE_OP_INSTANCE_FUNC(xir_op_type,            \
                                                       class_name)             \
  extern "C" {                                                                 \
  vart::cpu::CPURegFunc& VART_STR_CONCAT(class_name, __COUNTER__) =            \
      vart::cpu::CPURegFunc::instance()->register_func(                        \
          std::string{xir_op_type},                                            \
          [](const xir::Subgraph* subg, const xir::Op* xir_op,                 \
             IMapTBs_t inputs, CPUTBPtr_t output) {                            \
            auto in0 =                                                         \
                xir_op->get_input_tensors("input")[0] -> get_data_type();      \
            auto in1 = (xir_op->get_input_tensors("input").size() == 1)        \
                           ? xir_op->get_input_ops("weights")[0]               \
                             -> get_output_tensor() -> get_data_type()         \
                           : xir_op->get_input_tensors("input")[1]             \
                             -> get_data_type();                               \
            auto bias = in1;                                                   \
            if (xir_op->get_input_tensors("bias").size() > 0) {                \
              bias = xir_op->get_input_tensor("bias")->get_data_type();        \
            }                                                                  \
            vart::cpu::CPUOPBase* op = nullptr;                                \
            if (in0.type == xir::DataType::FLOAT && in0.bit_width == 32 &&     \
                in1.type == xir::DataType::FLOAT && in1.bit_width == 32 &&     \
                bias.type == xir::DataType::FLOAT && bias.bit_width == 32) {   \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<float, float, float>>(    \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 64 &&                                  \
                       in1.type == xir::DataType::FLOAT &&                     \
                       in1.bit_width == 32 &&                                  \
                       bias.type == xir::DataType::FLOAT &&                    \
                       bias.bit_width == 64) {                                 \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<double, float, double>>(  \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 32 &&                                  \
                       in1.type == xir::DataType::FLOAT &&                     \
                       in1.bit_width == 64 &&                                  \
                       bias.type == xir::DataType::FLOAT &&                    \
                       bias.bit_width == 64) {                                 \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<float, double, double>>(  \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 64 &&                                  \
                       in1.type == xir::DataType::FLOAT &&                     \
                       in1.bit_width == 64 &&                                  \
                       bias.type == xir::DataType::FLOAT &&                    \
                       bias.bit_width == 64) {                                 \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<double, double, double>>( \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type != xir::DataType::FLOAT &&                     \
                       in0.type != xir::DataType::BFLOAT &&                    \
                       in1.type != xir::DataType::FLOAT &&                     \
                       in1.type != xir::DataType::BFLOAT &&                    \
                       bias.type != xir::DataType::FLOAT &&                    \
                       bias.type != xir::DataType::BFLOAT) {                   \
              op = CPUOPFactory::Instance()                                    \
                       .create<                                                \
                           vart::cpu::class_name<int32_t, int32_t, int32_t>>(  \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type != xir::DataType::FLOAT &&                     \
                       in0.type != xir::DataType::BFLOAT &&                    \
                       in1.type == xir::DataType::FLOAT &&                     \
                       in1.bit_width == 32 &&                                  \
                       bias.type == xir::DataType::FLOAT &&                    \
                       bias.bit_width == 32) {                                 \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<int32_t, float, float>>(  \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 32 &&                                  \
                       in1.type != xir::DataType::FLOAT &&                     \
                       in1.type != xir::DataType::BFLOAT &&                    \
                       bias.type == xir::DataType::FLOAT &&                    \
                       bias.bit_width == 32) {                                 \
              op = CPUOPFactory::Instance()                                    \
                       .create<vart::cpu::class_name<float, int32_t, float>>(  \
                           subg, xir_op, inputs, output);                      \
            } else if (in0.type != xir::DataType::FLOAT &&                     \
                       in0.type != xir::DataType::BFLOAT &&                    \
                       in1.type == xir::DataType::FLOAT &&                     \
                       in1.bit_width == 64 &&                                  \
                       bias.type == xir::DataType::FLOAT &&                    \
                       bias.bit_width == 64) {                                 \
              op =                                                             \
                  CPUOPFactory::Instance()                                     \
                      .create<vart::cpu::class_name<int32_t, double, double>>( \
                          subg, xir_op, inputs, output);                       \
            } else if (in0.type == xir::DataType::FLOAT &&                     \
                       in0.bit_width == 64 &&                                  \
                       in1.type != xir::DataType::FLOAT &&                     \
                       in1.type != xir::DataType::BFLOAT &&                    \
                       bias.type == xir::DataType::FLOAT &&                    \
                       bias.bit_width == 64) {                                 \
              op =                                                             \
                  CPUOPFactory::Instance()                                     \
                      .create<vart::cpu::class_name<double, int32_t, double>>( \
                          subg, xir_op, inputs, output);                       \
            } else {                                                           \
              UNI_LOG_FATAL(VART_NOT_SUPPORT)                                  \
                  << "Not support " << #class_name << "<" << in0.to_string()   \
                  << ", " << in1.to_string() << ", " << bias.to_string()       \
                  << ">" << endl;                                              \
            }                                                                  \
            if (VART_DEBUG) {                                                  \
              UNI_LOG_DEBUG_INFO << "Construct vart::cpu::" << #class_name     \
                                 << "<" << in0.to_string() << ", "             \
                                 << in1.to_string() << ", "                    \
                                 << bias.to_string() << "> obj" << endl;       \
            }                                                                  \
            return op;                                                         \
          });                                                                  \
  }

#define REG_NONTP_OP_INSTANCE_FUNC(xir_op_type, class_name)                    \
  extern "C" {                                                                 \
  vart::cpu::CPURegFunc& VART_STR_CONCAT(class_name, __COUNTER__) =            \
      vart::cpu::CPURegFunc::instance()->register_func(                        \
          std::string{xir_op_type},                                            \
          [](const xir::Subgraph* subg, const xir::Op* xir_op,                 \
             IMapTBs_t inputs, CPUTBPtr_t output) {                            \
            vart::cpu::CPUOPBase* op =                                         \
                CPUOPFactory::Instance().create<vart::cpu::class_name>(        \
                    subg, xir_op, inputs, output);                             \
            if (VART_DEBUG) {                                                  \
              UNI_LOG_DEBUG_INFO << "Construct vart::cpu::" << #class_name     \
                                 << "obj" << endl;                             \
            }                                                                  \
            return op;                                                         \
          });                                                                  \
  }

namespace vart {
namespace cpu {

class CPUOPBase;

class CPUOPFactory {
 private:
  CPUOPFactory() = default;
  ~CPUOPFactory() = default;

 public:
  static CPUOPFactory& Instance() {
    static CPUOPFactory f;
    return f;
  }

  template <typename C>
  CPUOPBase* create(const xir::Subgraph* subg, const xir::Op* xir_op,
                    IMapTBs_t inputs, CPUTBPtr_t output) {
    // NOTE: must create every time, can not reuse,
    // because inputs and output will change
    auto op = make_unique<C>(subg, xir_op, inputs, output);
    if (op_map_.count(xir_op) == 0) {
      op_map_[xir_op] = std::move(op);
    }
    return op_map_.at(xir_op).get();
  }

 private:
  std::unordered_map<const xir::Op*, unique_ptr<CPUOPBase>> op_map_;
};

class CPURegFunc {
 public:
  using RegFuncType = function<CPUOPBase*(const xir::Subgraph*, const xir::Op*,
                                          IMapTBs_t, CPUTBPtr_t)>;

 private:
  explicit CPURegFunc();
  ~CPURegFunc() = default;

 public:
  static CPURegFunc* instance() {
    static CPURegFunc rf;

    return &rf;
  }

  CPURegFunc& register_func(const string& op_type, RegFuncType regfunc);
  RegFuncType get_register_func(const string& op_type);
  bool is_register(const string& op_type);

  void show_regmap();

 private:
  map<string, RegFuncType> regfunc_;
};

}  // namespace cpu
}  // namespace vart
