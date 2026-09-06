// Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <map>
#include <string>

namespace GraphEngine
{
  /**
   * Set of key-value pairs for application's quality-of-service parameters
   *
   * Supported keys:
   * "fps" :      frames per second
   * "latency" :  latency requirement in milliseconds
   * "priority"
   *
   */
  using qos_type = std::map<std::string, uint32_t>;

  enum class qos_priority {
	  qos_priority_realtime = 0x100,
	  qos_priority_high = 0x180,
	  qos_priority_normal = 0x200,
	  qos_priority_low = 0x280 // not supported
  };

  enum class performance_preference {
    perf_pref_default = 0,
    perf_pref_high_efficiency = 1
  };
}
