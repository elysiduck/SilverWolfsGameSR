// Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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
#include <string>
#include <memory>
#include "vart/runner.hpp"
#include "xir/graph/graph.hpp"

namespace GraphEngine
{
  /**
   * Construct a DpuRunner object from an XMODEL file.
   *
   * @param xmodel
   *  File path of XMODEL from which to construct this graph runner.
   * @param attrs
   *  Pointer to xir::Attrs. This is used for sharing metadata across all runners in the graph.
   * @return
   *  Unique Ptr to the polymorphic base class vart::Runner from which DpuRunner derives
   *
   * NOT IMPLEMENTED NOW, BECAUSE IT IS UNCLEAR WHICH DPU SUBGRAPH SHOULD BE SELECED 
   */
  std::unique_ptr<vart::Runner> create_dpu_runner(const std::string &xmodel, xir::Attrs* attrs = nullptr);

  /**
   * Construct a DpuRunner object from an xir::Graph.
   *
   * @param graph
   *  Pointer to xir::Graph.
   * @param attrs
   *  Pointer to xir::Attrs. This is used for sharing metadata across all runners in the graph.
   * @return
   *  Unique Ptr to the polymorphic base class vart::Runner from which DpuRunner derives
   *
   * NOT IMPLEMENTED NOW, BECAUSE IT IS UNCLEAR WHICH DPU SUBGRAPH SHOULD BE SELECED 
   */
  std::unique_ptr<vart::Runner> create_dpu_runner(const xir::Graph *graph, xir::Attrs* attrs = nullptr);

  /**
   * Construct a DpuRunner object from a dpu subgraph.
   *
   * @param root
   *  Pointer to the root subgraph inside of an XMODEL.
   * @param attrs
   *  Pointer to xir::Attrs. This is used for sharing metadata across all runners in the graph.
   * @return
   *  Unique Ptr to the polymorphic base class vart::Runner from which DpuRunner derives
   */
  std::unique_ptr<vart::Runner> create_dpu_runner(const xir::Subgraph *root, xir::Attrs* attrs = nullptr);

}
