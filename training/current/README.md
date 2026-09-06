# GameSR Training Code v1.7.1

PC-side end-to-end training and deployment preparation for 720p -> 1080p GameSR.

The training machine handles:

`clean -> benchmark -> full-image training -> NPU core ONNX -> AMD Quark XINT8/QDQ -> deployment bundle`

The handheld does not use this training environment. It only receives the model bundle and uses the separate standalone NPU benchmark/runtime.

Start with `先看这里-新手步骤.txt`. Technical details are in
`README_V1_7_1.md`. Legacy v1.7 design details remain in `README_V1_7.md`.
