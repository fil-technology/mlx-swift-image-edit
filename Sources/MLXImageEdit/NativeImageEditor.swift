import Foundation
import MLX
import Hub
#if canImport(CoreGraphics)
import CoreGraphics
#endif

// Narrow, esh-independent public surface for native instruction-driven (content-preserving) image editing.
// No esh/Studio types. Actor-contained; loads once from a local model directory and warm-reuses the model.

public struct InstructImageEditRequest: Sendable {
    public let prompt: String
    public let image: CGImage
    public let negativePrompt: String
    public let steps: Int
    public let guidanceScale: Float        // follow-instruction (default 7.0)
    public let imageGuidanceScale: Float   // preserve-source (default 1.5)
    public let seed: UInt64
    public init(prompt: String, image: CGImage, negativePrompt: String = "",
                steps: Int = 20, guidanceScale: Float = 7.0, imageGuidanceScale: Float = 1.5,
                seed: UInt64 = 0) {
        self.prompt = prompt; self.image = image; self.negativePrompt = negativePrompt
        self.steps = steps; self.guidanceScale = guidanceScale
        self.imageGuidanceScale = imageGuidanceScale; self.seed = seed
    }
}

public struct InstructImageEditProgress: Sendable {
    public enum Stage: Sendable { case preparing, loading, encoding, denoising, decoding, done }
    public let stage: Stage
    public let step: Int
    public let totalSteps: Int
}

public protocol NativeImageEditor: Sendable {
    func edit(_ request: InstructImageEditRequest,
              progress: @Sendable (InstructImageEditProgress) -> Void) async throws -> CGImage
}

public enum ImageEditError: Error, LocalizedError {
    case modelUnavailable(String)
    case encodeFailed
    public var errorDescription: String? {
        switch self {
        case .modelUnavailable(let s): return "instruct-pix2pix model unavailable: \(s)"
        case .encodeFailed: return "failed to encode output image"
        }
    }
}

/// Native InstructPix2Pix editor. Weights must be staged in the HubApi layout under `hubDownloadBase`:
/// `<hubDownloadBase>/models/timbrooks/instruct-pix2pix/{unet,vae,text_encoder,tokenizer,scheduler}/...`.
public actor InstructPix2PixEditor: NativeImageEditor {
    private let hubDownloadBase: URL?
    private let float16: Bool
    private let cacheLimitBytes: Int
    private let clearCacheAfterEdit: Bool
    private var sd: InstructPix2Pix?

    /// - Parameters:
    ///   - cacheLimitBytes: Upper bound on the MLX buffer cache. Left unbounded, MLX grows its reuse cache
    ///     to tens of GB across edits, which triggers memory pressure/swap and ~4× latency degradation.
    ///     Bounding it (default 2 GB) holds peak RSS near the true working set (~2 GB for SD1.5 fp16) with
    ///     no quality change — viable on 16 GB devices. 0 disables the bound.
    ///   - clearCacheAfterEdit: Return cached buffers to the OS after each edit. Default false: the cache cap
    ///     already bounds peak, and keeping the (bounded) cache warm makes back-to-back edits faster. Set true
    ///     for a memory-tight app that edits only occasionally and wants the lowest idle footprint.
    /// - Parameter hubDownloadBase: base for the HubApi cache layout; weights are read from
    ///   `<hubDownloadBase>/models/timbrooks/instruct-pix2pix/...`. `nil` uses HubApi's default
    ///   (`~/Documents/huggingface`). Pass the same value the weights were staged under.
    public init(hubDownloadBase: URL?, float16: Bool = true,
                cacheLimitBytes: Int = 2 * 1_073_741_824, clearCacheAfterEdit: Bool = false) {
        self.hubDownloadBase = hubDownloadBase
        self.float16 = float16
        self.cacheLimitBytes = cacheLimitBytes
        self.clearCacheAfterEdit = clearCacheAfterEdit
    }

    private func loaded(_ progress: @Sendable (InstructImageEditProgress) -> Void) throws -> InstructPix2Pix {
        if let sd { return sd }
        if cacheLimitBytes > 0 { GPU.set(cacheLimit: cacheLimitBytes) }
        progress(.init(stage: .loading, step: 0, totalSteps: 0))
        let hub = HubApi(downloadBase: hubDownloadBase, useOfflineMode: false)
        let config = instructPix2PixConfiguration()
        guard let model = try config.factory(hub, config,
              LoadConfiguration(float16: float16, quantize: false)) as? InstructPix2Pix else {
            throw ImageEditError.modelUnavailable("factory did not produce an InstructPix2Pix")
        }
        model.ensureLoaded()
        sd = model
        return model
    }

    public func edit(_ request: InstructImageEditRequest,
                     progress: @Sendable (InstructImageEditProgress) -> Void) async throws -> CGImage {
        progress(.init(stage: .preparing, step: 0, totalSteps: request.steps))
        let model = try loaded(progress)

        // Source → [H,W,3] float in [-1,1].
        progress(.init(stage: .encoding, step: 0, totalSteps: request.steps))
        let src = (Image(image: request.image).data.asType(.float32) / 255) * 2 - 1

        let params = InstructPix2Pix.EditParameters(
            prompt: request.prompt, negativePrompt: request.negativePrompt, steps: request.steps,
            guidanceScale: request.guidanceScale, imageGuidanceScale: request.imageGuidanceScale,
            seed: request.seed)

        let xt = try model.editLatents(
            source: src, params: params,
            onStep: { s, t in progress(.init(stage: .denoising, step: s, totalSteps: t)) },
            isCancelled: { Task.isCancelled })

        progress(.init(stage: .decoding, step: request.steps, totalSteps: request.steps))
        let decoded = model.decode(xt: xt)          // [1,h,w,3] in [0,1]
        let pixels = (decoded[0] * 255).asType(.uint8)
        let cg = Image(pixels).asCGImage()
        if clearCacheAfterEdit { GPU.clearCache() }
        progress(.init(stage: .done, step: request.steps, totalSteps: request.steps))
        return cg
    }
}
