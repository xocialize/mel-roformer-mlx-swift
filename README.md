# mel-roformer-mlx-swift

Swift / [MLX](https://github.com/ml-explore/mlx-swift) implementation of [Mel-Band-RoFormer](https://arxiv.org/abs/2310.01809) for vocal source separation on Apple Silicon.

Companion to the upstream Python implementation in [`Blaizzy/mlx-audio`](https://github.com/Blaizzy/mlx-audio) (currently in [PR #654](https://github.com/Blaizzy/mlx-audio/pull/654)). Loads pre-converted MLX weight checkpoints from HuggingFace and runs vocal separation natively on M-series Macs — no PyTorch, no CoreML, no CUDA.

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/xocialize/mel-roformer-mlx-swift.git", branch: "main"),
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "SwiftRoFormer", package: "mel-roformer-mlx-swift"),
        ]
    ),
]
```

Requirements: Swift 6.0+, macOS 15+ (Apple Silicon).

## Quick start

```swift
import SwiftRoFormer

// One-shot Hub download + weight load. The bundled config.json's
// `checkpoint_family` field auto-resolves the correct preset.
let model = try await MelRoFormer.fromPretrained(
    "mlx-community/mel-roformer-kim-vocal-2-mlx"
)

// Forward pass: input is (batch, channels, samples) at 44100 Hz.
// Returns separated vocals at the same shape.
let vocals = model(audio)

// Derive instrumental as (mixture - vocals) if needed.
let instrumental = audio - vocals
```

For lower-level control (explicit configuration, pre-downloaded weights, custom Hub clients):

```swift
let model = MelRoFormer(config: .kimVocal2)
try WeightLoader.loadWeights(into: model, from: localWeightsURL)
```

## Available checkpoints

Two parity-tested checkpoints are published on the [`mlx-community`](https://huggingface.co/mlx-community) HuggingFace org and can be loaded directly via `fromPretrained(_:)`:

| Preset | HuggingFace repo | Precision | Size | Parity SDR vs PyTorch |
|---|---|---|---:|---:|
| `.kimVocal2` | [`mlx-community/mel-roformer-kim-vocal-2-mlx`](https://huggingface.co/mlx-community/mel-roformer-kim-vocal-2-mlx) | bf16 | 435 MB | 66.08 dB |
| `.zfturboVocalsV1` | [`mlx-community/mel-roformer-zfturbo-vocals-v1-mlx`](https://huggingface.co/mlx-community/mel-roformer-zfturbo-vocals-v1-mlx) | fp16 | 64 MB | 44.19 dB |

Both are MIT-licensed. Both achieve the > 40 dB SDR target the upstream parity test enforces (treated as effectively bit-exact up to floating-point precision). See the [Mel-Band-RoFormer (MLX) Collection](https://huggingface.co/collections/mlx-community/mel-band-roformer-mlx-69ed0e92010230ba2795e30e) on HuggingFace for the discovery page.

A note on dtype: the smaller ZFTurbo architecture (`dim=192`, ~33M params) is bf16-sensitive — bf16 quantization drops parity to 21.96 dB on the same harness. fp16 (10-bit mantissa vs bf16's 7) recovers parity at the same file size, so `vocals_v1` ships as fp16. The wider Kim Vocal 2 (`dim=384`, ~228M params) absorbs bf16 truncation cleanly and ships as bf16.

## Architecture

```
audio (B, 2, T)
   │
   │ STFT (n_fft=2048, hop=441 or 512 depending on preset)
   ▼
complex spectrogram → CaC interleave → band split (60 mel bands)
   │
   ▼
N × DualAxisTransformer
   │  (each block: attention along time, then along freq)
   │  RoFormerAttention with RoPE, to_gates (bias=True)
   │  RoFormerFFN with SwiGLU
   │  RMSNorm with F.normalize semantics
   │
   ▼
MaskEstimator (per-band MLP with GLU gating)
   │  Complex multiply with input spectrogram
   ▼
iSTFT → separated vocals (B, 2, T)
```

The model is single-stem and outputs vocals only. Derive an instrumental track via `mixture - vocals`.

| Preset | dim | depth | hop | mask_depth | Params | Weight license |
|--------|----:|------:|----:|-----------:|-------:|----------------|
| `.kimVocal2`        | 384 | 6 | 441 | 2 | 228M | MIT (KimberleyJSN, relicensed Apr 2026) |
| `.zfturboVocalsV1`  | 192 | 8 | 512 | 1 |  33M | MIT (ZFTurbo v1.0.0) |

Two more presets (`viperx_vocals`, `zfturbo_bs_roformer`) are defined in the upstream Python port but not yet wired as Swift static properties — adding them is straightforward (matching hyperparameters in `RoFormerConfiguration`) but requires a corresponding licensed checkpoint to be useful.

## Public API

```swift
// Construct
let model = MelRoFormer()                                   // default = .kimVocal2
let model = MelRoFormer(config: .zfturboVocalsV1)
let model = try await MelRoFormer.fromPretrained("repo-id") // Hub-aware

// Run
let vocals = model(audio)                                   // (B, 2, T) → (B, 2, T)

// Load weights manually
try WeightLoader.loadWeights(into: model, from: weightsURL)
```

`MelRoFormer.fromPretrained(_:)` reads the published `config.json`'s `checkpoint_family` field to dispatch the matching preset automatically. To override (e.g. for a custom-trained checkpoint), pass `configuration:` explicitly.

## Testing

```bash
swift test
```

Unit tests cover configuration presets, weight-tree integrity, GPU forward-pass shape, and `fromPretrained` config dispatching. Tests requiring weight files (`WeightLoadingTests`) skip cleanly if `weights/<file>.safetensors` isn't present locally — download the safetensors from one of the `mlx-community` repos above to enable them.

If you hit `Failed to load the default metallib` on SPM CLI, it's the standard MLX-Swift Metal-shader issue. Build the project once from Xcode to populate `mlx-swift_Cmlx.bundle` in DerivedData, then copy it next to the test executable:

```bash
cp -R "$HOME/Library/Developer/Xcode/DerivedData/<project>/Build/Products/Debug/mlx-swift_Cmlx.bundle" \
   .build/arm64-apple-macosx/debug/
```

## Converting other checkpoints

For checkpoints not yet on `mlx-community` (`viperx_vocals`, `anvuew`, custom-trained variants), use the upstream Python converter from [`Blaizzy/mlx-audio`](https://github.com/Blaizzy/mlx-audio) (pending merge of [PR #654](https://github.com/Blaizzy/mlx-audio/pull/654)):

```bash
python -m mlx_audio.sts.models.mel_roformer.convert \
    --input path/to/model.ckpt \
    --output ./weights/ \
    --preset zfturbo_vocals_v1 \
    --dtype bfloat16   # or float16 / float32
```

The output is a content-addressed `.safetensors` plus a companion `.config.json`. Drop both into a directory and pass to `WeightLoader.loadWeights`, or upload to HuggingFace and use `fromPretrained(_:)`.

License obligations transfer to the user upon redistribution — the converter emits source-aware warnings for known checkpoint origins (`KimberleyJSN`, `ZFTurbo`, `TRvlvr`/viperx, `anvuew`). Don't redistribute checkpoints you don't have license clearance for.

## Provenance and parity

Architecture lineage:

- [`lucidrains/BS-RoFormer`](https://github.com/lucidrains/BS-RoFormer) — original PyTorch implementation (MIT)
- [`ZFTurbo/Music-Source-Separation-Training`](https://github.com/ZFTurbo/Music-Source-Separation-Training) — training-time configurations (MIT)
- [`Blaizzy/mlx-audio`](https://github.com/Blaizzy/mlx-audio) — Python MLX port (Apache-2.0, [PR #654](https://github.com/Blaizzy/mlx-audio/pull/654))

Five subtle bugs were caught and fixed during parity validation against the PyTorch reference:

- RoPE convention (interleaved-pair rotation, not halved-split)
- RMSNorm `eps` semantics (`F.normalize` style, not `1e-5` additive)
- `to_gates` bias=True
- `MaskEstimator` configurable depth (1 for `zfturbo_vocals_v1`, 2 for others)
- RMSNorm parameter naming (`gamma` ↔ `weight`)

These are the documented fragile points in the lucidrains → MLX path; the upstream parity test in `mlx-audio` enforces them.

## License

This package is MIT-licensed. See [`LICENSE`](LICENSE).

The Mel-Band-RoFormer architecture is MIT (lucidrains BS-RoFormer + ZFTurbo MSS-Training). Individual checkpoint licenses are tracked per-preset (see the table above and the model cards on `mlx-community`).

## Citation

If you use this package, please cite the Mel-Band-RoFormer paper:

```bibtex
@misc{lu2023melband,
  title         = {Mel-Band {RoFormer} for Music Source Separation},
  author        = {Lu, Wei-Tsung and Wang, Ju-Chiang and Won, Minz and Choi, Keunwoo and Song, Xuchen},
  year          = {2023},
  eprint        = {2310.01809},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2310.01809}
}
```
