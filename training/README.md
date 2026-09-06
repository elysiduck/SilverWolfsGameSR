# Training

`current/` is the consolidated GameSR training v1.8.2 source. It includes streaming, persistent RAM cache, EMA resume, diagnostics, game-aware training and late-polish/information-aware training.

Virtual environments, datasets, checkpoints, previews, TensorBoard logs and exported models are external artifacts.

`BUILD-GAMESR-OUTPUT-CHANNEL-PROBES.cmd` and `tools/generate_output_channel_probes.py` generate the C16-L4 O3/O4/O6/O8 XINT8 payload used by the handheld output-channel probe. This is an architecture-cost probe, not a training-quality model.

`BUILD-GAMESR-POTENTIAL4-OUTPUT-PROBES.cmd`, `tools/generate_potential4_output_probes.py` and `config/potential_backbones.json` generate the next focused set: C8-L5/C16-L4/C24-L3/C32-L3, each at O3/O4/O6/O8. It preserves a common seed within each backbone so output-width comparisons remain clean.
