# Output-channel probe v1.0.2 release

This release has three assets:

| Asset | SHA256 | Purpose |
| --- | --- | --- |
| `GameSR-XDNA-v1.3-OutputChannelProbe-Patch-v1.0.2-HandheldExtract-FIX.zip` | `2D9B4CD8B3F8256E264ECAF8AC1845C1D98242037B5C6A3BFCD007E11C788993` | Training-side generator and handheld Runner v1.0.2. |
| `GameSR-NPU-Output-Channel-Probe-Models-4-20260906-121057.zip` | `B88813B03E455C98BEC0F93A2F887E839EB6F2E80EBD650237F36209A5C29E0C` | Prebuilt O3/O4/O6/O8 XINT8 payload. |
| `GameSR-NPU-Output-Channel-Probe-20260906-121534-COMPACT.zip` | `4C8A3999802B971432813738B00485C7B9E1C4758FD50FD821983992DD4457E5` | Result table, JSON summaries and launch logs. |

The v1.0.2 Runner fixes Windows PowerShell 5.1 extraction failures by using a short `_ocp` cache and a temporary extraction directory. It does not change model-generation methodology.
