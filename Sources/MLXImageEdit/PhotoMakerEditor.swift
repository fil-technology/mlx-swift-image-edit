import Foundation
import MLX
import Hub
#if canImport(CoreGraphics)
import CoreGraphics
#endif

// Public provider surface for native PhotoMaker v1 identity-preserving image editing. This is a DISCRETE
// provider (sibling to InstructPix2PixEditor), not a replacement for `image.edit`: PhotoMaker preserves
// facial identity; STYLE stays an independent prompt/preset concern (no style hardcoded here). Actor-
// contained; loads once from local weights + a staged SDXL base and warm-reuses the pipeline.

public struct IdentityImageEditRequest: Sendable {
    /// The reference face(s) whose identity to preserve (Phase 1: a single face).
    public let referenceImage: CGImage
    /// Class word for the subject, e.g. "woman", "man", "person" — the identity token PhotoMaker fuses into.
    public let subjectWord: String
    /// Style / scene description, e.g. "a high-quality 3D animated film character, soft cinematic lighting".
    /// Product-facing style presets map to this string; nothing here is hardcoded.
    public let stylePrompt: String
    public let negativePrompt: String
    public let steps: Int
    /// Steps before switching from text-only to id-fused conditioning (identity strength vs. editability).
    public let startMergeStep: Int
    public let guidanceScale: Float
    public let seed: UInt64
    /// Square output side in pixels (SDXL micro-conditioning is set to match). Default 1024.
    public let size: Int

    public init(referenceImage: CGImage, subjectWord: String, stylePrompt: String,
                negativePrompt: String = "photorealistic, realistic photo, blurry, deformed, low quality",
                steps: Int = 40, startMergeStep: Int = 10, guidanceScale: Float = 5.0,
                seed: UInt64 = 0, size: Int = 1024) {
        self.referenceImage = referenceImage; self.subjectWord = subjectWord
        self.stylePrompt = stylePrompt; self.negativePrompt = negativePrompt
        self.steps = steps; self.startMergeStep = startMergeStep
        self.guidanceScale = guidanceScale; self.seed = seed; self.size = size
    }

    /// The composed PhotoMaker prompt: the trigger word `img` must follow the class word.
    var composedPrompt: String { "a \(subjectWord) img \(stylePrompt)" }
}

public enum PhotoMakerEditError: Error, LocalizedError {
    case modelUnavailable(String)
    public var errorDescription: String? {
        switch self { case .modelUnavailable(let s): return "PhotoMaker model unavailable: \(s)" }
    }
}

/// Native PhotoMaker v1 identity editor. Weights: converted ID encoder + folded UNet-LoRA deltas (see the
/// conversion step) plus an SDXL base staged in the HubApi layout under `hubDownloadBase`.
public actor PhotoMakerV1Editor: NativeImageEditor {
    private let hubDownloadBase: URL?
    private let idEncoderPath: String
    private let loraPath: String
    private let styleLoraPath: String?
    private let styleScale: Float
    private let cacheLimitBytes: Int
    private var pm: PhotoMakerV1?

    /// - Parameters:
    ///   - styleLoraPath: optional independent style preset (a commercially-licensed style LoRA, converted to
    ///     diffusers-keyed deltas). Style is decoupled from identity — swap the preset without touching the
    ///     identity provider. `nil` = base SDXL (no style LoRA).
    ///   - styleScale: strength of the style preset (0…~1.2).
    public init(hubDownloadBase: URL?, idEncoderPath: String, loraPath: String,
                styleLoraPath: String? = nil, styleScale: Float = 1.0,
                cacheLimitBytes: Int = 4 * 1_073_741_824) {
        self.hubDownloadBase = hubDownloadBase
        self.idEncoderPath = idEncoderPath
        self.loraPath = loraPath
        self.styleLoraPath = styleLoraPath
        self.styleScale = styleScale
        self.cacheLimitBytes = cacheLimitBytes
    }

    private func loaded(_ progress: @Sendable (InstructImageEditProgress) -> Void) throws -> PhotoMakerV1 {
        if let pm { return pm }
        if cacheLimitBytes > 0 { GPU.set(cacheLimit: cacheLimitBytes) }
        progress(.init(stage: .loading, step: 0, totalSteps: 0))
        let hub = HubApi(downloadBase: hubDownloadBase, useOfflineMode: false)
        let model = try PhotoMakerV1(hub: hub, configuration: photoMakerSDXLConfiguration(),
                                     dType: .float16, loraPath: loraPath, idEncoderPath: idEncoderPath,
                                     styleLoraPath: styleLoraPath, styleScale: styleScale)
        model.ensureLoaded()
        pm = model
        return model
    }

    /// NativeImageEditor conformance: maps the generic instruct request onto an identity edit. The prompt is
    /// used as the style text and `subjectWord` defaults to "person"; callers wanting explicit control use
    /// `edit(_ request: IdentityImageEditRequest, ...)`.
    public func edit(_ request: InstructImageEditRequest,
                     progress: @Sendable (InstructImageEditProgress) -> Void) async throws -> CGImage {
        try edit(IdentityImageEditRequest(
            referenceImage: request.image, subjectWord: "person", stylePrompt: request.prompt,
            negativePrompt: request.negativePrompt, steps: request.steps,
            guidanceScale: request.guidanceScale, seed: request.seed), progress: progress)
    }

    public func edit(_ request: IdentityImageEditRequest,
                     progress: @Sendable (InstructImageEditProgress) -> Void) throws -> CGImage {
        progress(.init(stage: .preparing, step: 0, totalSteps: request.steps))
        let model = try loaded(progress)
        progress(.init(stage: .encoding, step: 0, totalSteps: request.steps))
        let idPixel = photoMakerPreprocess(request.referenceImage)
        let cond = model.makeConditioning(prompt: request.composedPrompt,
                                          negative: request.negativePrompt, idPixel: idPixel)
        let xt = try model.generateLatents(
            conditioning: cond, steps: request.steps, startMergeStep: request.startMergeStep,
            guidanceScale: request.guidanceScale, seed: request.seed, size: request.size,
            onStep: { s, t in progress(.init(stage: .denoising, step: s, totalSteps: t)) },
            isCancelled: { Task.isCancelled })
        progress(.init(stage: .decoding, step: request.steps, totalSteps: request.steps))
        let decoded = model.decodeTiled(xt)
        let pixels = (decoded[0] * 255).asType(.uint8)
        let cg = Image(pixels).asCGImage()
        progress(.init(stage: .done, step: request.steps, totalSteps: request.steps))
        return cg
    }
}
