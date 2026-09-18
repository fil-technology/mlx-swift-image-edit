import Testing
import Foundation
@testable import MLXImageEdit

// Pure-Swift (metal-free) unit tests. The `swift test` CLI runner cannot load the MLX metallib, so the
// tensor-level numerics (channel concat, dual-CFG) are verified in the working MLX context by the
// `ip2p-validate selftest` executable — the same assertions, executed where Metal is available. The full
// pixel-level gate (instruction following + source preservation) is the `ip2p-validate` on-device run.

@Suite struct RequestSurfaceTests {

    @Test func requestDefaultsAreInstructEditTuned() {
        let r = InstructImageEditRequest(prompt: "make it red", image: tinyCG())
        #expect(r.steps == 20)
        #expect(r.guidanceScale == 7.0)          // follow-instruction
        #expect(r.imageGuidanceScale == 1.5)     // preserve-source
        #expect(r.negativePrompt == "")
        #expect(r.seed == 0)
    }

    @Test func requestOverridesArePreserved() {
        let r = InstructImageEditRequest(prompt: "p", image: tinyCG(), negativePrompt: "blurry",
                                         steps: 30, guidanceScale: 9, imageGuidanceScale: 2.2, seed: 99)
        #expect(r.steps == 30 && r.guidanceScale == 9 && r.imageGuidanceScale == 2.2)
        #expect(r.negativePrompt == "blurry" && r.seed == 99)
    }

    @Test func modelFileLayoutIsFP16SD15() {
        // The exact file set the editor loads and the download stages (fp16, SD1.5/CLIP-L layout).
        let f = Set(instructPix2PixModelFiles)
        #expect(f.contains("unet/diffusion_pytorch_model.fp16.safetensors"))
        #expect(f.contains("text_encoder/model.fp16.safetensors"))
        #expect(f.contains("vae/diffusion_pytorch_model.fp16.safetensors"))
        #expect(f.contains("scheduler/scheduler_config.json"))
        #expect(f.contains("tokenizer/vocab.json") && f.contains("tokenizer/merges.txt"))
    }
}

#if canImport(CoreGraphics)
import CoreGraphics
private func tinyCG() -> CGImage {
    let cs = CGColorSpaceCreateDeviceRGB()
    let ctx = CGContext(data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 0,
                        space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.setFillColor(CGColor(red: 0, green: 0, blue: 1, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
    return ctx.makeImage()!
}
#endif
