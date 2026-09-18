import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Darwin
import MLX
import MLXImageEdit

setbuf(stdout, nil)  // stream results live even when redirected to a file

func mlxMem() -> String {
    let s = GPU.snapshot()
    func gb(_ b: Int) -> String { String(format: "%.2f", Double(b) / 1_073_741_824) }
    return "active=\(gb(s.activeMemory))G cache=\(gb(s.cacheMemory))G peak=\(gb(s.peakMemory))G"
}

func footprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) { p in
        p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
}
func loadCG(_ path: String) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}
func saveCG(_ cg: CGImage, _ path: String) {
    guard let d = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                  UTType.png.identifier as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(d, cg, nil); CGImageDestinationFinalize(d)
}

// args: <hubBase> <outDir> "<imgPath>::<prompt>::<name>" ...   OR   selftest
let a = CommandLine.arguments

// Deterministic numeric self-test, run in the working MLX/Metal context (the `swift test` CLI cannot load
// the MLX metallib; these are the same assertions as MLXImageEditTests, executed where MLX is available).
if a.count >= 2, a[1] == "selftest" {
    var pass = 0, fail = 0
    func check(_ name: String, _ cond: Bool) { if cond { pass += 1; print("ok   \(name)") } else { fail += 1; print("FAIL \(name)") } }

    let noisy = MLXArray(0 ..< 16).reshaped(1, 2, 2, 4).asType(.float32)
    let cond = MLXArray.ones([1, 2, 2, 4])
    let cc = IP2PMath.channelConcat(noisy, cond)
    check("channelConcat shape==[1,2,2,8]", cc.shape == [1, 2, 2, 8])
    check("channelConcat ch0==noisy", cc[0, 0, 0, 0].item(Float.self) == 0)
    check("channelConcat ch4==cond", cc[0, 0, 0, 4].item(Float.self) == 1)

    let g = IP2PMath.combineGuidance(text: MLXArray([10, 10] as [Float]), image: MLXArray([4, 6] as [Float]),
                                     uncond: MLXArray([1, 2] as [Float]), guidanceScale: 7, imageGuidanceScale: 1.5)
    check("combineGuidance ch0==47.5", abs(g[0].item(Float.self) - 47.5) < 1e-4)
    check("combineGuidance ch1==36.0", abs(g[1].item(Float.self) - 36.0) < 1e-4)
    let z = IP2PMath.combineGuidance(text: MLXArray([9] as [Float]), image: MLXArray([3] as [Float]),
                                     uncond: MLXArray([5] as [Float]), guidanceScale: 0, imageGuidanceScale: 0)
    check("combineGuidance zero-scales==uncond", z[0].item(Float.self) == 5)

    check("model files include unet fp16", instructPix2PixModelFiles.contains("unet/diffusion_pytorch_model.fp16.safetensors"))
    check("model files include text_encoder fp16", instructPix2PixModelFiles.contains("text_encoder/model.fp16.safetensors"))
    check("model files include vae fp16", instructPix2PixModelFiles.contains("vae/diffusion_pytorch_model.fp16.safetensors"))

    let cs = CGColorSpaceCreateDeviceRGB()
    let ctx = CGContext(data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 0,
                        space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    let r = InstructImageEditRequest(prompt: "p", image: ctx.makeImage()!)
    check("request defaults steps==20", r.steps == 20)
    check("request defaults gScale==7", r.guidanceScale == 7.0)
    check("request defaults imgScale==1.5", r.imageGuidanceScale == 1.5)

    print("SELFTEST pass=\(pass) fail=\(fail)")
    exit(fail == 0 ? 0 : 1)
}

guard a.count >= 4 else { print("usage: ip2p-validate <hubBase> <outDir> <img::prompt::name> ...  |  selftest"); exit(2) }
let hubBase = URL(fileURLWithPath: a[1]); let outDir = a[2]
try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
struct Job { let img: String, prompt: String, name: String }
let jobs: [Job] = a[3...].compactMap {
    let p = $0.components(separatedBy: "::"); guard p.count == 3 else { return nil }
    return Job(img: p[0], prompt: p[1], name: p[2])
}

// Optional memory controls (env): IP2P_CACHE_MB caps the MLX buffer cache; IP2P_CLEAR=1 clears it per edit.
if let mb = ProcessInfo.processInfo.environment["IP2P_CACHE_MB"].flatMap(Int.init) {
    GPU.set(cacheLimit: mb * 1_048_576); print("SET cacheLimit=\(mb)MB")
}
let clearPerEdit = ProcessInfo.processInfo.environment["IP2P_CLEAR"] == "1"

let editor = InstructPix2PixEditor(hubDownloadBase: hubBase)
print("MEM baseline=\(String(format: "%.0f", footprintMB()))MB  jobs=\(jobs.count)  clearPerEdit=\(clearPerEdit)")

@discardableResult
func run(_ job: Job, seed: UInt64 = 42) async -> Double {
    guard let src = loadCG(job.img) else { print("SRC_ERR \(job.img)"); return -1 }
    let t0 = Date()
    do {
        let out = try await editor.edit(
            .init(prompt: job.prompt, image: src, steps: 20, guidanceScale: 7.0, imageGuidanceScale: 1.5, seed: seed),
            progress: { p in FileHandle.standardError.write("  [\(job.name)] \(p.stage) \(p.step)/\(p.totalSteps)\n".data(using: .utf8)!) })
        let dt = Date().timeIntervalSince(t0)
        saveCG(out, "\(outDir)/\(job.name).png")
        print("EDIT_OK \(job.name) '\(job.prompt)' time=\(String(format: "%.1f", dt))s rss=\(String(format: "%.0f", footprintMB()))MB mlx[\(mlxMem())] -> \(job.name).png")
        if clearPerEdit { GPU.clearCache() }
        return dt
    } catch { print("EDIT_ERR \(job.name): \(error)"); return -1 }
}

var peak = 0.0
await { () async in
    // 1) cold edit (includes first model load)
    let tCold = await run(jobs[0])
    peak = max(peak, footprintMB())
    print("FIRST_EDIT_COLD(incl model load)=\(String(format: "%.1f", tCold))s peak=\(String(format: "%.0f", peak))MB")
    // 2) remaining edits warm (model already loaded)
    for j in jobs.dropFirst() { await run(j); peak = max(peak, footprintMB()) }
    // 3) warm reuse: repeat job0 on the loaded model
    let tWarm = await run(jobs[0], seed: 7)
    print("WARM_EDIT(reuse)=\(String(format: "%.1f", tWarm))s")
    // 4) cancellation: start an edit, cancel mid-flight, expect CancellationError + no PNG
    let cancelName = "cancel_probe"
    let ct = Task { await run(Job(img: jobs[0].img, prompt: "make it red", name: cancelName)) }
    try? await Task.sleep(nanoseconds: 500_000_000)
    ct.cancel()
    _ = await ct.value
    let saved = FileManager.default.fileExists(atPath: "\(outDir)/\(cancelName).png")
    print("CANCEL: pngSaved=\(saved) (expect false for a clean cancel)")
    peak = max(peak, footprintMB())
    print("PEAK_MEM=\(String(format: "%.0f", peak))MB")
}()
