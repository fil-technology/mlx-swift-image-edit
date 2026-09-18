# mlx-swift-image-edit

Native, content-preserving **instruction** image editing (InstructPix2Pix) on Apple silicon via
[MLX](https://github.com/ml-explore/mlx-swift). Given a source image and an instruction
("turn the car red", "make it a sunset", "add snow"), it edits the image while preserving the
subject and scene — distinct from img2img "restyle" that regenerates from the prompt.

- **Public API:** `InstructPix2PixEditor` (an `actor`) + `InstructImageEditRequest` / `InstructImageEditProgress`.
- **Model:** `timbrooks/instruct-pix2pix`, fp16 safetensors (~2 GB), token-free, CreativeML OpenRAIL-M.
  Weights are read from an on-disk HubApi layout you stage; the package does not redistribute them.
- **Memory:** the editor caps the MLX buffer cache at 2 GB by default (peak RSS ~7 GB for SD1.5 fp16),
  which keeps it viable on 16 GB devices — see [VALIDATION.md](VALIDATION.md).
- **Dependencies:** `mlx-swift` (0.29.x) + `swift-transformers` `Hub` (1.0.x). No swift-syntax.

```swift
let editor = InstructPix2PixEditor(hubDownloadBase: hubURL)   // <hubURL>/models/timbrooks/instruct-pix2pix/...
let out = try await editor.edit(
    .init(prompt: "turn the car red", image: sourceCGImage),
    progress: { p in print(p.stage, p.step, "/", p.totalSteps) })
```

On-device validation (M1 Pro): 3/3 instruction edits preserve the source; see `VALIDATION.md` and
`ip2p-validate` (a CLI harness + `selftest`). Licensing/attribution: see `NOTICE`.
