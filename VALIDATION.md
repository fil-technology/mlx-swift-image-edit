# mlx-swift-image-edit — Validation Gate

Standalone, esh-independent Swift package for **native, content-preserving instruction image editing**
(InstructPix2Pix) on Apple Silicon via MLX. This document records the real, on-device validation gate.

**Status: PASS** (quality) + a memory finding investigated and fixed in-package.
Scope note: this is the standalone-package gate only. It does **not** modify esh or Esh Studio, and no RC is
published. Integration is a separate, later step.

---

## 1. What was validated

- **Model:** `timbrooks/instruct-pix2pix`, fp16 safetensors, **2.0 GB** total, token-free (no HF login),
  license **CreativeML OpenRAIL-M** (commercial use permitted with the use-based restrictions; self-hostable).
- **Weights integrity (sha256, pinned & verified OK):**
  - `unet/diffusion_pytorch_model.fp16.safetensors` `0d6bbc0a95dd125196d327a660b43d24c56f433eb30d2776f1327fb86bd38f78` (1.6 GB)
  - `text_encoder/model.fp16.safetensors` `77795e2023adcf39bc29a884661950380bd093cf0750a966d473d1718dc9ef4e` (235 MB)
  - `vae/diffusion_pytorch_model.fp16.safetensors` `4fbcf0ebe55a0984f5a5e00d8c4521d52359af7229bb4d81890039d2aa16dd7c` (160 MB)
- **Architecture confirmed from configs:** UNet `in_channels=8` (IP2P), `cross_attention_dim=768`;
  CLIP-L text encoder (`hidden_size=768`); `EulerAncestralDiscreteScheduler`, `epsilon` prediction.
- **Host:** Apple M1 Pro, 32 GB unified memory. Weights on external SSD (HubApi layout); build scratch on
  internal APFS.

## 2. Pixel-level gate (mandatory) — PASS

Real source photos (Wikimedia Commons, CC), center-cropped to 512×512. Each edit uses the shipped defaults
(20 steps, textGuidance 7.0, imageGuidance 1.5, seed 42). Both dimensions were evaluated: **instruction
following** and **source preservation**.

| Test | Instruction | Instruction followed? | Source preserved? |
|------|-------------|-----------------------|-------------------|
| Blue car → red | "turn the car red" | Yes — car is red | Strong — same car body/shape, wheels, chrome trim, building, doorways, road all intact |
| Day → sunset | "make it a sunset" | Yes — warm golden sky | Strong — same cloud structure, foliage, grass, tree line, left-side plant unchanged |
| Add snow | "add snow" | Yes — snow-covered ground, wintered grass | Strong — same composition, plant, tree line, sky, distant tower |

Contact sheet (source | edit) delivered to the user: `ip2p_validation_sheet.png`.
This is genuine content-preserving instruct editing — **not** SDXL-Turbo-style regenerate-from-prompt
(which does not preserve subject/scene and is correctly kept as `image.restyle`, a separate capability).

## 3. Performance measurements

20 steps, 512×512, warm model reuse across edits.

| Configuration | Peak RSS | MLX active (working set) | Cold edit (incl. model load) | Warm edit |
|---------------|----------|--------------------------|------------------------------|-----------|
| Unbounded MLX cache | **24.3 GB** | — | 58 s | **123–140 s** (degrading) |
| **Cap 2 GB, keep warm (shipped default)** | **6.9 GB** | 2.14 GB | ~65 s | **32–37 s** |
| Cap 2 GB + clear each edit (opt-in) | 6.3–6.8 GB | 2.14 GB | ~59 s | ~33–66 s |

- **First model load:** ~2 GB fp16 read from SSD; folded into the ~59–65 s cold edit.
- **Warm reuse:** confirmed — the model stays resident in the actor and subsequent edits skip loading.
- **Cancellation:** cooperative (`Task.isCancelled` checked each denoise step) — raises `CancellationError`
  promptly and writes **no** output. Verified: `pngSaved=false`.
- Absolute latency shows machine-state variance (thermal / concurrent load) on a shared dev machine;
  the memory numbers are stable and reproducible.

## 4. Key finding: unbounded MLX buffer cache (investigated + fixed in-package)

The first run peaked at **24.3 GB** and warm edits *degraded* to 123–140 s — the classic signature of
memory pressure/swap. Instrumenting `GPU.snapshot()` showed the true working set is only **2.14 GB**
(SD1.5 fp16 weights + 3-branch activations); the rest was MLX's unbounded reuse cache.

**Fix, baked into `InstructPix2PixEditor` as the default:** `GPU.set(cacheLimit: 2 GB)` on load. This holds
peak RSS at ~6.9 GB with **no change to output pixels** and removes the latency degradation (warm edits
~32–37 s). Result: viable on 16 GB devices, comfortable on 32 GB. An opt-in `clearCacheAfterEdit` returns
the bounded cache to the OS between edits for memory-tight, edit-occasionally apps.

## 5. Numerics (as implemented)

- **8-channel UNet input** per branch: `[noisy latent (4ch) ++ conditioning latent (4ch)]` on the channel
  axis. Source latent = `VAE.encode(source).mean / scalingFactor`, concatenated **clean at every step**
  (this, not img2img noising, is what preserves subject/scene). Uncond branch uses a zero image latent.
- **Dual classifier-free guidance** (Brooks et al. 2023):
  `eps = uncond + textGuidance·(text − image) + imageGuidance·(image − uncond)`.
- **Conditioning:** prompt and negative are tokenized **together** so both pad to a common sequence length
  (required for the 3-branch concat); CLIP-L last hidden state, SD1.5-correct.
- Sampler: `SimpleEulerAncestralSampler` (matches the model's `EulerAncestralDiscreteScheduler`).

## 6. Tests

- **`swift test`** (pure-Swift, metal-free): 3/3 pass — request defaults/overrides, fp16 SD1.5 file layout.
- **`ip2p-validate selftest`** (MLX numerics, run where the metallib loads): 12/12 pass — 8-channel concat
  shape/ordering, dual-CFG formula (incl. zero-scale collapse), file layout, request defaults.
- Note: the `swift test` CLI runner cannot load the MLX metallib (a known mlx-swift limitation; works in
  Xcode), so tensor-level numerics run via the `selftest` executable instead of xctest.

## 7. Dependency coexistence

Deps: `mlx-swift` (upToNextMinor 0.29.1) + `swift-transformers` (upToNextMinor 1.0.0, Hub only).
**No swift-syntax**, no mlx-swift 0.31.x. This is coexistence-safe with esh (mlx-swift 0.29.x) and Esh
Studio's LLM.swift (swift-syntax 602.x) — the reason InstructPix2Pix/SD1.5 was chosen over Qwen-Image-Edit
for the first native instruct-edit path.

## 8. How to reproduce

```
# stage weights (HubApi layout) at <HUB>/models/timbrooks/instruct-pix2pix/{unet,text_encoder,vae,scheduler,tokenizer}
swift build -c release --product ip2p-validate
.build/release/ip2p-validate selftest
.build/release/ip2p-validate <HUB> <outDir> "src.png::turn the car red::car_red"
```

Vendored StableDiffusion source: `ml-explore/mlx-swift-examples @ 9bff95ca` (2.29.1), MIT — see
`Sources/MLXImageEdit/StableDiffusion/VENDORED_FROM.txt`.
