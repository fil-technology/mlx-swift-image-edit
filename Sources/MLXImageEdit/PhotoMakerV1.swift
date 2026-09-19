import Foundation
import MLX
import MLXNN
import MLXRandom
import Hub
#if canImport(CoreGraphics)
import CoreGraphics
#endif

// Native PhotoMaker v1 (Apache-2.0 / OpenCLIP, no InsightFace) on the vendored SDXL pipeline (mlx-swift
// 0.29.x). Identity-preserving instruction edit: an SDXL base UNet with the PhotoMaker LoRA folded in, the
// PhotoMaker ID encoder (see PhotoMakerIDEncoder.swift), and a two-phase (`startMergeStep`) denoise that uses
// text-only conditioning first, then id-fused conditioning. Style stays a prompt/preset concern — nothing
// here hardcodes a "3D style". This is a sibling provider to InstructPix2Pix; IP2P is untouched.

/// SDXL-base file layout (fp16) the loader reads from the HubApi cache under `<hub>/models/<id>/...`.
func photoMakerSDXLConfiguration(id: String = "stabilityai/stable-diffusion-xl-base-1.0")
    -> StableDiffusionConfiguration {
    StableDiffusionConfiguration(
        id: id,
        files: [
            .unetConfig: "unet/config.json",
            .unetWeights: "unet/diffusion_pytorch_model.fp16.safetensors",
            .textEncoderConfig: "text_encoder/config.json",
            .textEncoderWeights: "text_encoder/model.fp16.safetensors",
            .textEncoderConfig2: "text_encoder_2/config.json",
            .textEncoderWeights2: "text_encoder_2/model.fp16.safetensors",
            .vaeConfig: "vae/config.json",
            .vaeWeights: "vae/diffusion_pytorch_model.fp16.safetensors",
            .diffusionConfig: "scheduler/scheduler_config.json",
            .tokenizerVocabulary: "tokenizer/vocab.json",
            .tokenizerMerges: "tokenizer/merges.txt",
            .tokenizerVocabulary2: "tokenizer_2/vocab.json",
            .tokenizerMerges2: "tokenizer_2/merges.txt",
        ],
        defaultParameters: { EvaluateParameters(cfgWeight: 5, steps: 40) },
        factory: { _, _, _ in throw PhotoMakerError.notLoadable })
}

enum PhotoMakerError: Error { case notLoadable }

/// Fold a compact low-rank LoRA (keys `<target>.lora_down` [r,in], `<target>.lora_up` [out,r],
/// `<target>.lora_scale`) into `raw` UNet weights (diffusers keys) natively: `W += userScale·scale·(up·down)`.
/// No Python/diffusers at runtime. Returns the number of folded targets.
@discardableResult
func foldCompactLoRA(_ raw: inout [String: MLXArray], url: URL, userScale: Float) throws -> Int {
    let f = try loadArrays(url: url)
    let targets = f.keys.filter { $0.hasSuffix(".lora_down") }.map { String($0.dropLast(".lora_down".count)) }
    var folded = 0
    for t in targets {
        guard let down = f["\(t).lora_down"], let up = f["\(t).lora_up"], let w = raw["\(t).weight"] else { continue }
        let scale = f["\(t).lora_scale"]?.item(Float.self) ?? 1.0
        let delta = up.asType(.float32).matmul(down.asType(.float32))       // [out,in]
        raw["\(t).weight"] = w.asType(.float32) + (userScale * scale) * delta
        folded += 1
    }
    return folded
}

/// Load the SDXL base UNet with the PhotoMaker attention LoRA folded in, plus an optional independent style
/// LoRA (e.g. a commercially-licensed 3D-render style) at a tunable strength. Both are compact low-rank
/// safetensors folded natively in Swift — style stays decoupled from identity.
func loadPhotoMakerUNetWithLoRA(
    hub: HubApi, configuration: StableDiffusionConfiguration, loraPath: String, dType: DType,
    styleLoraPath: String? = nil, styleScale: Float = 1.0
) throws -> UNetModel {
    let unetConfig = try loadConfiguration(
        hub: hub, configuration: configuration, key: .unetConfig, type: UNetConfiguration.self)
    let model = UNetModel(configuration: unetConfig)

    let unetURL = resolve(hub: hub, configuration: configuration, key: .unetWeights)
    var raw = try loadArrays(url: unetURL)                                   // diffusers keys, fp16
    let pmFolded = try foldCompactLoRA(&raw, url: URL(fileURLWithPath: loraPath), userScale: 1.0)
    precondition(pmFolded > 0, "PhotoMaker LoRA: folded 0 targets")
    if let styleLoraPath, styleScale != 0 {
        let sFolded = try foldCompactLoRA(&raw, url: URL(fileURLWithPath: styleLoraPath), userScale: styleScale)
        precondition(sFolded > 0, "style LoRA: folded 0 targets")
    }
    let weights = raw.flatMap { unetRemap(key: $0.key, value: $0.value.asType(dType)) }
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
    return model
}

final class PhotoMakerV1: StableDiffusion {
    let textEncoder2: CLIPTextModel
    let tokenizer2: CLIPTokenizer
    let idEncoder: PhotoMakerIDEncoder

    init(hub: HubApi, configuration: StableDiffusionConfiguration, dType: DType,
         loraPath: String, idEncoderPath: String,
         styleLoraPath: String? = nil, styleScale: Float = 1.0) throws {
        let dc = try loadDiffusionConfiguration(hub: hub, configuration: configuration)
        let unet = try loadPhotoMakerUNetWithLoRA(
            hub: hub, configuration: configuration, loraPath: loraPath, dType: dType,
            styleLoraPath: styleLoraPath, styleScale: styleScale)
        self.textEncoder2 = try loadTextEncoder(
            hub: hub, configuration: configuration, configKey: .textEncoderConfig2,
            weightsKey: .textEncoderWeights2, dType: dType)
        self.tokenizer2 = try loadTokenizer(
            hub: hub, configuration: configuration, vocabulary: .tokenizerVocabulary2,
            merges: .tokenizerMerges2)
        self.idEncoder = try loadPhotoMakerIDEncoder(path: idEncoderPath, dType: .float32)
        try super.init(
            hub: hub, configuration: configuration, dType: dType, diffusionConfiguration: dc,
            unet: unet, sampler: SimpleEulerSampler(configuration: dc))
    }

    override func ensureLoaded() {
        super.ensureLoaded()
        eval(textEncoder2, idEncoder)
    }

    // MARK: - Tokenization

    /// Build a length-77 token id sequence with the trigger word "img" removed and the preceding class token
    /// marked. Mirrors PhotoMaker's encode_prompt_with_trigger_word for num_tokens = 1.
    private func triggerIdsAndMask(_ prompt: String, tokenizer: CLIPTokenizer, padId: Int32)
        -> (ids: [Int32], mask: [Bool], classIndex: Int) {
        let imgId = Int32(tokenizer.vocabulary["img</w>"] ?? -1)
        var clean: [Int32] = []
        var classIndex = -1
        for tid in tokenizer.tokenize(text: prompt) {
            if tid == imgId { classIndex = clean.count - 1 } else { clean.append(tid) }
        }
        var ids = clean
        if ids.count > 77 { ids = Array(ids.prefix(77)) }
        else { ids += Array(repeating: padId, count: 77 - ids.count) }
        let mask = (0 ..< 77).map { $0 == classIndex }         // num_tokens = 1 → single class token
        return (ids, mask, classIndex)
    }

    private func plainIds(_ text: String, tokenizer: CLIPTokenizer, padId: Int32) -> [Int32] {
        var ids = tokenizer.tokenize(text: text)
        if ids.count > 77 { ids = Array(ids.prefix(77)) }
        else { ids += Array(repeating: padId, count: 77 - ids.count) }
        return ids
    }

    private func encode(ids1: [Int32], ids2: [Int32]) -> (embeds: MLXArray, pooled: MLXArray) {
        let o1 = textEncoder(MLXArray(ids1).reshaped(1, 77))
        let o2 = textEncoder2(MLXArray(ids2).reshaped(1, 77))
        let h1 = o1.hiddenStates.dropLast().last!               // [1,77,768] penultimate
        let h2 = o2.hiddenStates.dropLast().last!               // [1,77,1280] penultimate
        return (concatenated([h1, h2], axis: -1), o2.pooledOutput)   // [1,77,2048], [1,1280]
    }

    /// Conditioning bundle for a PhotoMaker generation.
    struct Conditioning {
        var textOnly: MLXArray      // [1,77,2048] pre-fuse
        var fused: MLXArray         // [1,77,2048] id-fused
        var pooled: MLXArray        // [1,1280]
        var negative: MLXArray      // [1,77,2048]
        var negativePooled: MLXArray  // [1,1280]
        var mask: MLXArray          // [1,77] class-token mask
    }

    /// `idPixel` is [1,1,3,224,224] CLIP-normalized. Returns text-only + id-fused conditioning + negatives.
    func makeConditioning(prompt: String, negative: String, idPixel: MLXArray) -> Conditioning {
        let eos = Int32(tokenizer.eosToken)                    // 49407, CLIP-L pad
        let (ids1, maskArr, _) = triggerIdsAndMask(prompt, tokenizer: tokenizer, padId: eos)
        let (ids2, _, _) = triggerIdsAndMask(prompt, tokenizer: tokenizer2, padId: 0)   // bigG pad = 0
        let (rawEmbeds, pooled) = encode(ids1: ids1, ids2: ids2)
        let mask = MLXArray(maskArr.map { $0 ? Int32(1) : 0 }).reshaped(1, 77)
        let fused = idEncoder(idPixel, promptEmbeds: rawEmbeds, classTokensMask: mask)

        let (nids1, nids2) = (plainIds(negative, tokenizer: tokenizer, padId: eos),
                              plainIds(negative, tokenizer: tokenizer2, padId: 0))
        let (negEmbeds, negPooled) = encode(ids1: nids1, ids2: nids2)
        return Conditioning(textOnly: rawEmbeds, fused: fused, pooled: pooled,
                            negative: negEmbeds, negativePooled: negPooled, mask: mask)
    }

    /// Memory-bounded VAE decode: decode the latent in overlapping tiles and blend with a separable
    /// triangular window. Keeps full-resolution quality while capping peak memory (the single full-frame
    /// fp32 decode is what drives the 22 GB peak) and avoids the Metal command-buffer watchdog on large ops.
    /// `latent` is [1,H,W,4]; returns [1,H*8,W*8,3] in [0,1].
    func decodeTiled(_ latent: MLXArray, tile: Int = 64, overlap: Int = 16) -> MLXArray {
        let H = latent.dim(1), W = latent.dim(2), s = 8
        if H <= tile && W <= tile { return decode(xt: latent) }   // small enough: one shot
        let stride = max(1, tile - overlap)
        func starts(_ n: Int) -> [Int] {
            var xs: [Int] = [], p = 0
            while true { let p0 = min(p, n - tile); xs.append(max(0, p0)); if p0 + tile >= n { break }; p += stride }
            return Array(Set(xs)).sorted()
        }
        func tri(_ n: Int) -> MLXArray {
            let idx = MLXArray(0 ..< Int32(n)).asType(.float32)
            return minimum(idx + 1, Float(n) - idx)               // triangular: ~1 at edges, peak center
        }
        var acc = MLXArray.zeros([1, H * s, W * s, 3])
        var wsum = MLXArray.zeros([1, H * s, W * s, 1])
        for y0 in starts(H) {
            let y1 = min(y0 + tile, H)
            for x0 in starts(W) {
                let x1 = min(x0 + tile, W)
                let zt = latent[0..., y0 ..< y1, x0 ..< x1, 0...]
                var dec = decode(xt: zt)                            // [1,(y1-y0)*8,(x1-x0)*8,3] in [0,1]
                eval(dec)
                let ph = (y1 - y0) * s, pw = (x1 - x0) * s
                let mask = (tri(ph).reshaped(1, ph, 1, 1) * tri(pw).reshaped(1, 1, pw, 1))  // [1,ph,pw,1]
                let py0 = y0 * s, py1 = y1 * s, px0 = x0 * s, px1 = x1 * s
                acc[0..., py0 ..< py1, px0 ..< px1, 0...] = acc[0..., py0 ..< py1, px0 ..< px1, 0...] + dec * mask
                wsum[0..., py0 ..< py1, px0 ..< px1, 0...] = wsum[0..., py0 ..< py1, px0 ..< px1, 0...] + mask
                GPU.clearCache()
                dec = acc                                          // release
            }
        }
        return acc / maximum(wsum, MLXArray(1e-6))
    }

    // MARK: - Generate

    /// Two-phase PhotoMaker denoise at `size`×`size` (SDXL micro-conditioning at that resolution). Returns the
    /// final latent; decode with `decode(xt:)`. `onStep(i,total)` per step; `isCancelled` checked cooperatively.
    func generateLatents(conditioning c: Conditioning, steps: Int, startMergeStep: Int,
                         guidanceScale: Float, seed: UInt64, size: Int,
                         onStep: (Int, Int) -> Void, isCancelled: () -> Bool) throws -> MLXArray {
        MLXRandom.seed(seed)
        let latent = size / 8
        var xt = sampler.samplePrior(shape: [1, latent, latent, autoencoder.latentChannels], dType: dType)
        let timeVals: [Float] = [Float(size), Float(size), 0, 0, Float(size), Float(size)]
        let timeIds = broadcast(MLXArray(timeVals).reshaped(1, -1), to: [2, 6]).asType(dType)

        let stepList = sampler.timeSteps(steps: steps, start: sampler.maxTime, dType: dType)
        let total = stepList.count
        for (i, (t, tPrev)) in stepList.enumerated() {
            if isCancelled() { throw CancellationError() }
            let merge = i > startMergeStep
            let cond = merge ? c.fused : c.textOnly
            let pooled = c.pooled                              // pooled is identical either phase (see note)
            let condCat = concatenated([c.negative, cond], axis: 0)          // [2,77,2048]
            let pooledCat = concatenated([c.negativePooled, pooled], axis: 0)  // [2,1280]
            let xtIn = concatenated([xt, xt], axis: 0)
            let tUnet = broadcast(t, to: [2])
            let eps = unet(xtIn, timestep: tUnet, encoderX: condCat, textTime: (pooledCat, timeIds))
            let parts = eps.split(parts: 2, axis: 0)
            let epsPred = parts[0] + guidanceScale * (parts[1] - parts[0])
            xt = sampler.step(epsPred: epsPred, xt: xt, t: t, tPrev: tPrev)
            eval(xt)
            onStep(i + 1, total)
        }
        return xt
    }
}

/// CLIP image preprocessing for the ID encoder: resize the shorter side to 224 (bicubic-ish), center-crop
/// 224×224, scale to [0,1], normalize with the CLIP mean/std. Returns [1,1,3,224,224].
func photoMakerPreprocess(_ cg: CGImage) -> MLXArray {
    let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    let std: [Float] = [0.26862954, 0.26130258, 0.27577711]
    // Center-crop to square, resample to 224 via CoreGraphics, read bytes directly (robust shapes).
    let side = 224
    let cs = CGColorSpaceCreateDeviceRGB()
    var buf = [UInt8](repeating: 0, count: side * side * 4)
    buf.withUnsafeMutableBytes { raw in
        let ctx = CGContext(data: raw.baseAddress, width: side, height: side, bitsPerComponent: 8,
                            bytesPerRow: side * 4, space: cs,
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        ctx.interpolationQuality = .high
        let w = cg.width, h = cg.height, m = min(w, h)
        let crop = cg.cropping(to: CGRect(x: (w - m) / 2, y: (h - m) / 2, width: m, height: m)) ?? cg
        ctx.draw(crop, in: CGRect(x: 0, y: 0, width: side, height: side))
    }
    let rgba = MLXArray(buf).reshaped(side, side, 4).asType(.float32) / 255   // [224,224,4]
    let rgb = rgba[0..., 0..., 0 ..< 3]                                        // [224,224,3]
    let meanA = MLXArray(mean).reshaped(1, 1, 3)
    let stdA = MLXArray(std).reshaped(1, 1, 3)
    let norm = (rgb - meanA) / stdA                                           // [224,224,3]
    return norm.transposed(2, 0, 1).reshaped(1, 1, 3, side, side)             // [1,1,3,224,224]
}

private func cosFlat(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.reshaped(-1).asType(.float32), y = b.reshaped(-1).asType(.float32)
    return ((x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))).item(Float.self)
}

/// Validate native SDXL conditioning (tokenization + both encoders + id fusion) against reference goldens.
public func photoMakerConditioningParityCheck(
    hubBase: String, idEncoderPath: String, loraPath: String, goldenPath: String
) throws -> String {
    let hub = HubApi(downloadBase: URL(fileURLWithPath: hubBase), useOfflineMode: false)
    let pm = try PhotoMakerV1(hub: hub, configuration: photoMakerSDXLConfiguration(), dType: .float16,
                              loraPath: loraPath, idEncoderPath: idEncoderPath)
    pm.ensureLoaded()
    let g = try loadArrays(url: URL(fileURLWithPath: goldenPath))
    let prompt = "a woman img transformed into a high-quality 3D animated film character, pixar style, preserving recognizable facial identity, same hairstyle, same smiling expression, same pose and composition, soft cinematic lighting, highly detailed"
    let neg = "photorealistic, realistic photo, blurry, deformed, low quality, grayscale, extra fingers"
    let c = pm.makeConditioning(prompt: prompt, negative: neg, idPixel: g["id_pixel_values"]!.asType(.float32))
    eval(c.textOnly, c.fused, c.pooled, c.negative, c.negativePooled)
    return "textOnly~trigger=\(cosFlat(c.textOnly, g["prompt_embeds_trigger"]!)) "
        + "textOnly~textOnlyRef=\(cosFlat(c.textOnly, g["prompt_embeds_text_only"]!)) "
        + "fused=\(cosFlat(c.fused, g["fused_prompt_embeds"]!)) "
        + "neg=\(cosFlat(c.negative, g["negative_prompt_embeds"]!)) "
        + "pooled=\(cosFlat(c.pooled, g["pooled_text_only"]!)) "
        + "negPooled=\(cosFlat(c.negativePooled, g["negative_pooled"]!))"
}

#if canImport(ImageIO) && canImport(CoreGraphics)
import ImageIO
import UniformTypeIdentifiers

private func pmLoadCG(_ path: String) -> CGImage? {
    guard let s = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(s, 0, nil)
}
private func pmSaveCG(_ cg: CGImage, _ path: String) {
    guard let d = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                  UTType.png.identifier as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(d, cg, nil); CGImageDestinationFinalize(d)
}
private func pmMem() -> String {
    let s = GPU.snapshot()
    func gb(_ b: Int) -> String { String(format: "%.2f", Double(b) / 1_073_741_824) }
    return "active=\(gb(s.activeMemory))G cache=\(gb(s.cacheMemory))G peak=\(gb(s.peakMemory))G"
}

/// End-to-end native PhotoMaker v1 validation gate: load once, generate the given seeds (warm reuse),
/// cancel one mid-flight, then generate again (post-cancel reuse). Saves `pm_seed<seed>.png` per seed and
/// returns a multi-line report of timings + MLX memory. Bounds the MLX cache (default 4 GB).
public func photoMakerRunGate(
    hubBase: String, idEncoderPath: String, loraPath: String, facePath: String, outDir: String,
    prompt: String, negativePrompt: String, seeds: [UInt64], steps: Int, startMergeStep: Int,
    guidanceScale: Float, size: Int, cacheLimitBytes: Int = 4 * 1_073_741_824,
    styleLoraPath: String? = nil, styleScale: Float = 1.0
) throws -> String {
    if cacheLimitBytes > 0 { GPU.set(cacheLimit: cacheLimitBytes) }
    try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    let hub = HubApi(downloadBase: URL(fileURLWithPath: hubBase), useOfflineMode: false)

    let tLoad = Date()
    let pm = try PhotoMakerV1(hub: hub, configuration: photoMakerSDXLConfiguration(), dType: .float16,
                              loraPath: loraPath, idEncoderPath: idEncoderPath,
                              styleLoraPath: styleLoraPath, styleScale: styleScale)
    pm.ensureLoaded()
    let loadT = Date().timeIntervalSince(tLoad)
    guard let cg = pmLoadCG(facePath) else { return "PM_GATE_ERR could not load \(facePath)" }
    let idPixel = photoMakerPreprocess(cg)

    func dbg(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }
    dbg("LOADED load=\(loadT)s; preprocessed idPixel=\(idPixel.shape)")
    var lines = ["firstLoad=\(String(format: "%.1f", loadT))s"]
    for (i, seed) in seeds.enumerated() {
        let cond = pm.makeConditioning(prompt: prompt, negative: negativePrompt, idPixel: idPixel)
        dbg("COND ok seed=\(seed) textOnly=\(cond.textOnly.shape) fused=\(cond.fused.shape)")
        let t0 = Date()
        let xt = try pm.generateLatents(conditioning: cond, steps: steps, startMergeStep: startMergeStep,
                                        guidanceScale: guidanceScale, seed: seed, size: size,
                                        onStep: { s, t in if s == 1 || s % 10 == 0 { dbg("  step \(s)/\(t)") } }, isCancelled: { false })
        if i == 0 { dbg("denoisePeak=\(String(format: "%.2f", Double(GPU.snapshot().peakMemory) / 1_073_741_824))G") }
        GPU.clearCache(); if i == 0 { GPU.resetPeakMemory() }     // free denoise cache before the VAE decode
        dbg("DENOISE done seed=\(seed)")
        let decoded = pm.decodeTiled(xt)
        let pixels = (decoded[0] * 255).asType(.uint8)
        eval(pixels)
        if i == 0 { dbg("decodePeak=\(String(format: "%.2f", Double(GPU.snapshot().peakMemory) / 1_073_741_824))G") }
        GPU.clearCache()
        pmSaveCG(Image(pixels).asCGImage(), "\(outDir)/pm_seed\(seed).png")
        let dt = Date().timeIntervalSince(t0)
        lines.append("seed=\(seed) \(i == 0 ? "firstGen" : "warmGen")=\(String(format: "%.1f", dt))s mlx[\(pmMem())] -> pm_seed\(seed).png")
    }

    if ProcessInfo.processInfo.environment["PM_LIGHT"] == "1" {
        return lines.joined(separator: "\n")     // sweep mode: seeds only, skip cancel/reuse
    }

    // Cancellation: cancel after 5 steps; expect CancellationError and no saved image.
    var calls = 0
    var cancelThrew = false
    do {
        _ = try pm.generateLatents(
            conditioning: pm.makeConditioning(prompt: prompt, negative: negativePrompt, idPixel: idPixel),
            steps: steps, startMergeStep: startMergeStep, guidanceScale: guidanceScale, seed: 999,
            size: size, onStep: { _, _ in }, isCancelled: { calls += 1; return calls > 5 })
    } catch is CancellationError { cancelThrew = true } catch {}
    lines.append("cancel: threw=\(cancelThrew) afterStepChecks=\(calls)")

    // Post-cancel reuse: generate again, must succeed.
    let tR = Date()
    let condR = pm.makeConditioning(prompt: prompt, negative: negativePrompt, idPixel: idPixel)
    let xtR = try pm.generateLatents(conditioning: condR, steps: steps, startMergeStep: startMergeStep,
                                     guidanceScale: guidanceScale, seed: seeds.first ?? 42, size: size,
                                     onStep: { _, _ in }, isCancelled: { false })
    let decR = pm.decodeTiled(xtR); let pxR = (decR[0] * 255).asType(.uint8); eval(pxR)
    pmSaveCG(Image(pxR).asCGImage(), "\(outDir)/pm_postcancel.png")
    lines.append("postCancelReuse=\(String(format: "%.1f", Date().timeIntervalSince(tR)))s ok mlx[\(pmMem())]")
    return lines.joined(separator: "\n")
}
#endif
