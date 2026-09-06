# Training

`current/` is the consolidated GameSR training v1.8.2 source. It includes streaming, persistent RAM cache, EMA resume, diagnostics, game-aware training and late-polish/information-aware training.

Virtual environments, datasets, checkpoints, previews, TensorBoard logs and exported models are external artifacts.

`BUILD-GAMESR-OUTPUT-CHANNEL-PROBES.cmd` and `tools/generate_output_channel_probes.py` generate the C16-L4 O3/O4/O6/O8 XINT8 payload used by the handheld output-channel probe. This is an architecture-cost probe, not a training-quality model.
