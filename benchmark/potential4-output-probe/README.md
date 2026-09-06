# Potential-4 output probe v1.1.0

This focused experiment selects four survivors from the 105-architecture sweep—C8-L5, C16-L4, C24-L3 and C32-L3—and evaluates output widths O3, O4, O6 and O8. Every comparison is against O3 of the **same** backbone. It measures pure NPU ONNX/VitisAI runtime cost only, not Magpie map/stitch/upload, decoder cost or image quality.

## Average NPU latency (ms)

| Backbone (C-L) | O3 | O4 | O6 | O8 |
| --- | ---: | ---: | ---: | ---: |
| **C8-L5** | 7.074 | 7.216 | **7.694** | 8.037 |
| **C16-L4** | **7.044** | **7.182** | **7.738** | **7.976** |
| **C24-L3** | 7.301 | 7.455 | 7.899 | 8.224 |
| **C32-L3** | 8.100 | 8.291 | 8.714 | 9.007 |

| Backbone | O3 | O4 delta | O6 delta | O8 delta |
| --- | ---: | ---: | ---: | ---: |
| C8-L5 | 7.0740 ms | +2.012% | +8.772% | +13.613% |
| C16-L4 | 7.0437 ms | +1.966% | +9.863% | +13.230% |
| C24-L3 | 7.3011 ms | +2.107% | +8.195% | +12.643% |
| C32-L3 | 8.1005 ms | +2.350% | +7.576% | +11.187% |

All 16 observations are `NPU_ONLY`. The compact result, individual summaries and launch logs are retained in `benchmark/results/potential4-output-probe-20260906/`.

`handheld/` must be copied into a complete Benchmark v1.3 root that already contains its runtime. Generate a fresh 16-model payload in the Quark-enabled training environment with `training/current/BUILD-GAMESR-POTENTIAL4-OUTPUT-PROBES.cmd` or `scripts/build-potential4-output-probes.ps1`.
