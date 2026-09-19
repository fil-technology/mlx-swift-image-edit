import Foundation
import MLX
import MLXNN

// PhotoMaker v1 ID encoder (Apache-2.0, OpenCLIP path — NO InsightFace). Ports TencentARC/PhotoMaker
// `photomaker/model.py` (PhotoMakerIDEncoder): a CLIP ViT-L/14 vision transformer whose pooled output is
// projected two ways (768 + 1280 → 2048, the SDXL text-embedding width) and fused into the prompt embeddings
// at the class-token position(s). Module parameter paths mirror the HF/PhotoMaker checkpoint keys so the
// converted safetensors loads directly (see photoMakerIDEncoderRemap).

private func quickGELU(_ x: MLXArray) -> MLXArray { x * sigmoid(1.702 * x) }

/// One CLIP vision transformer encoder layer (pre-norm, no causal mask).
final class CLIPVisionEncoderLayer: Module {
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: CLIPVisionAttention
    @ModuleInfo(key: "mlp") var mlp: CLIPVisionMLP

    init(dim: Int, heads: Int, mlpDim: Int, eps: Float) {
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: dim, eps: eps)
        self._layerNorm2.wrappedValue = LayerNorm(dimensions: dim, eps: eps)
        self._selfAttn.wrappedValue = CLIPVisionAttention(dim: dim, heads: heads)
        self._mlp.wrappedValue = CLIPVisionMLP(dim: dim, mlpDim: mlpDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + selfAttn(layerNorm1(x))
        h = h + mlp(layerNorm2(h))
        return h
    }
}

final class CLIPVisionAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let heads: Int
    let scale: Float

    init(dim: Int, heads: Int) {
        self.heads = heads
        self.scale = 1.0 / Float(dim / heads).squareRoot()
        self._qProj.wrappedValue = Linear(dim, dim)
        self._kProj.wrappedValue = Linear(dim, dim)
        self._vProj.wrappedValue = Linear(dim, dim)
        self._outProj.wrappedValue = Linear(dim, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, N, D) = (x.dim(0), x.dim(1), x.dim(2))
        let hd = D / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped(B, N, heads, hd).transposed(0, 2, 1, 3) }
        let q = split(qProj(x)) * scale
        let k = split(kProj(x))
        let v = split(vProj(x))
        let attn = softmax(q.matmul(k.transposed(0, 1, 3, 2)), axis: -1)
        let out = attn.matmul(v).transposed(0, 2, 1, 3).reshaped(B, N, D)
        return outProj(out)
    }
}

final class CLIPVisionMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    init(dim: Int, mlpDim: Int) {
        self._fc1.wrappedValue = Linear(dim, mlpDim)
        self._fc2.wrappedValue = Linear(mlpDim, dim)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(quickGELU(fc1(x))) }
}

final class CLIPVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(dim: Int, patch: Int, positions: Int) {
        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: dim, kernelSize: IntOrPair(patch),
            stride: IntOrPair(patch), bias: false)
        self._classEmbedding.wrappedValue = MLXArray.zeros([dim])
        self._positionEmbedding.wrappedValue = Embedding(embeddingCount: positions, dimensions: dim)
    }

    /// `pixel` is NCHW in [-1..1]-ish CLIP-normalized space; converted to NHWC for the MLX conv.
    func callAsFunction(_ pixel: MLXArray) -> MLXArray {
        let B = pixel.dim(0)
        let nhwc = pixel.transposed(0, 2, 3, 1)
        var patches = patchEmbedding(nhwc)                       // [B, gh, gw, dim]
        let dim = patches.dim(3)
        patches = patches.reshaped(B, -1, dim)                  // [B, gh*gw, dim]
        let cls = broadcast(classEmbedding.reshaped(1, 1, dim), to: [B, 1, dim])
        var x = concatenated([cls, patches], axis: 1)           // [B, 1+gh*gw, dim]
        x = x + positionEmbedding.weight[.newAxis]
        return x
    }
}

final class CLIPVisionTransformer: Module {
    @ModuleInfo(key: "embeddings") var embeddings: CLIPVisionEmbeddings
    @ModuleInfo(key: "pre_layrnorm") var preLayrnorm: LayerNorm   // HF spelling
    @ModuleInfo(key: "encoder") var encoder: CLIPVisionEncoder
    @ModuleInfo(key: "post_layernorm") var postLayernorm: LayerNorm

    init(dim: Int, layers: Int, heads: Int, mlpDim: Int, patch: Int, positions: Int, eps: Float) {
        self._embeddings.wrappedValue = CLIPVisionEmbeddings(dim: dim, patch: patch, positions: positions)
        self._preLayrnorm.wrappedValue = LayerNorm(dimensions: dim, eps: eps)
        self._encoder.wrappedValue = CLIPVisionEncoder(dim: dim, layers: layers, heads: heads, mlpDim: mlpDim, eps: eps)
        self._postLayernorm.wrappedValue = LayerNorm(dimensions: dim, eps: eps)
    }

    /// Returns the pooled output (post-layernorm of the CLS token) — PhotoMaker's `vision_model(...)[1]`.
    func callAsFunction(_ pixel: MLXArray) -> MLXArray {
        var x = preLayrnorm(embeddings(pixel))
        x = encoder(x)
        let pooled = x[0..., 0, 0...]         // [B, dim]  (CLS token)
        return postLayernorm(pooled)
    }
}

final class CLIPVisionEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [CLIPVisionEncoderLayer]
    init(dim: Int, layers: Int, heads: Int, mlpDim: Int, eps: Float) {
        self._layers.wrappedValue = (0 ..< layers).map { _ in
            CLIPVisionEncoderLayer(dim: dim, heads: heads, mlpDim: mlpDim, eps: eps)
        }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for l in layers { x = l(x) }
        return x
    }
}

/// PhotoMaker FuseModule MLP: layernorm → fc1 → gelu → fc2 (+ optional residual).
final class PMFuseMLP: Module {
    @ModuleInfo var layernorm: LayerNorm
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    let useResidual: Bool
    init(inDim: Int, hiddenDim: Int, outDim: Int, useResidual: Bool) {
        self.useResidual = useResidual
        self._layernorm.wrappedValue = LayerNorm(dimensions: inDim)
        self._fc1.wrappedValue = Linear(inDim, hiddenDim)
        self._fc2.wrappedValue = Linear(hiddenDim, outDim)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = layernorm(x)
        h = fc2(gelu(fc1(h)))          // reference uses exact nn.GELU()
        return useResidual ? h + residual : h
    }
}

final class PMFuseModule: Module {
    @ModuleInfo var mlp1: PMFuseMLP
    @ModuleInfo var mlp2: PMFuseMLP
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    init(embedDim: Int) {
        self._mlp1.wrappedValue = PMFuseMLP(inDim: embedDim * 2, hiddenDim: embedDim, outDim: embedDim, useResidual: false)
        self._mlp2.wrappedValue = PMFuseMLP(inDim: embedDim, hiddenDim: embedDim, outDim: embedDim, useResidual: true)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: embedDim)
    }
    /// fuse one class-token embedding row with one id embedding row: [.., 2048] each → [.., 2048].
    func fuse(promptRow: MLXArray, idRow: MLXArray) -> MLXArray {
        var stacked = concatenated([promptRow, idRow], axis: -1)   // [.., 4096]
        stacked = mlp1(stacked) + promptRow
        stacked = mlp2(stacked)
        return layerNorm(stacked)
    }
}

/// The PhotoMaker v1 ID encoder. Fuses a reference-face embedding into the prompt embeddings at the class
/// token. Phase 1 implements the single-class-token / single-ID-image case (num_tokens = 1), which is exactly
/// what the validated reference uses; multi-token / multi-subject is a later generalization.
final class PhotoMakerIDEncoder: Module {
    @ModuleInfo(key: "vision_model") var visionModel: CLIPVisionTransformer
    @ModuleInfo(key: "visual_projection") var visualProjection: Linear
    @ModuleInfo(key: "visual_projection_2") var visualProjection2: Linear
    @ModuleInfo(key: "fuse_module") var fuseModule: PMFuseModule

    override init() {
        // CLIP ViT-L/14: dim 1024, 24 layers, 16 heads, mlp 4096, patch 14, 257 positions.
        self._visionModel.wrappedValue = CLIPVisionTransformer(
            dim: 1024, layers: 24, heads: 16, mlpDim: 4096, patch: 14, positions: 257, eps: 1e-5)
        self._visualProjection.wrappedValue = Linear(1024, 768, bias: false)
        self._visualProjection2.wrappedValue = Linear(1024, 1280, bias: false)
        self._fuseModule.wrappedValue = PMFuseModule(embedDim: 2048)
    }

    /// - Parameters:
    ///   - idPixelValues: [b, n, 3, 224, 224] CLIP-normalized reference faces (Phase 1: n = 1).
    ///   - promptEmbeds: [b, 77, 2048] SDXL prompt embeddings (with the trigger word).
    ///   - classTokensMask: [b, 77] with exactly one `1` per batch (num_tokens = 1).
    /// - Returns: [b, 77, 2048] prompt embeddings with the class token replaced by the fused id embedding.
    func callAsFunction(_ idPixelValues: MLXArray, promptEmbeds: MLXArray, classTokensMask: MLXArray) -> MLXArray {
        let b = idPixelValues.dim(0)
        let n = idPixelValues.dim(1)
        let pixel = idPixelValues.reshaped(b * n, idPixelValues.dim(2), idPixelValues.dim(3), idPixelValues.dim(4))
        let shared = visionModel(pixel)                                  // [b*n, 1024]
        let idEmbeds = concatenated([visualProjection(shared), visualProjection2(shared)], axis: -1)  // [b*n, 2048]
        let idRow = idEmbeds.reshaped(b, n, -1)[0..., 0, 0...]           // [b, 2048]  (Phase 1: n == 1)

        let mask = classTokensMask.asType(promptEmbeds.dtype)[0..., 0..., .newAxis]  // [b,77,1]
        let classRow = (promptEmbeds * mask).sum(axis: 1)               // [b, 2048] (single class token)
        let fusedRow = fuseModule.fuse(promptRow: classRow, idRow: idRow.asType(promptEmbeds.dtype))  // [b,2048]
        return promptEmbeds * (1 - mask) + fusedRow[0..., .newAxis, 0...] * mask
    }
}

/// Remap PhotoMaker checkpoint keys → `PhotoMakerIDEncoder` parameter paths. Keys already match (module keys
/// mirror the HF layout); the only value transform is the patch-embedding conv NCHW→NHWC.
func photoMakerIDEncoderRemap(key: String, value: MLXArray) -> [(String, MLXArray)] {
    var value = value
    if key == "vision_model.embeddings.patch_embedding.weight", value.ndim == 4 {
        value = value.transposed(0, 2, 3, 1)                            // [out,in,kh,kw] → [out,kh,kw,in]
        value = value.reshaped(-1).reshaped(value.shape)               // ensure contiguous
    }
    return [(key, value)]
}

import Hub

/// Load a PhotoMaker v1 ID encoder from a converted safetensors file (fp32).
func loadPhotoMakerIDEncoder(path: String, dType: DType = .float32) throws -> PhotoMakerIDEncoder {
    let enc = PhotoMakerIDEncoder()
    try loadWeights(url: URL(fileURLWithPath: path), model: enc, mapper: photoMakerIDEncoderRemap, dType: dType)
    eval(enc)
    return enc
}

/// Numeric parity check of the native ID encoder against the reference-captured golden intermediates.
/// Returns a one-line report (max/mean abs error + cosine vs the reference fused prompt embeddings).
public func photoMakerIDEncoderParityCheck(idEncoderPath: String, goldenPath: String) throws -> String {
    let enc = try loadPhotoMakerIDEncoder(path: idEncoderPath)
    let g = try loadArrays(url: URL(fileURLWithPath: goldenPath))
    guard let pixel = g["id_pixel_values"], let pe = g["prompt_embeds_trigger"],
          let ctm = g["class_tokens_mask"], let want = g["fused_prompt_embeds"] else {
        return "MISSING golden tensors: \(g.keys.sorted())"
    }
    let out = enc(pixel.asType(.float32), promptEmbeds: pe.asType(.float32), classTokensMask: ctm)
    eval(out)
    let wantF = want.asType(.float32)
    let diff = abs(out - wantF)
    let maxd = diff.max().item(Float.self)
    let meand = diff.mean().item(Float.self)
    let a = out.reshaped(-1), b = wantF.reshaped(-1)
    let cos = ((a * b).sum() / (sqrt((a * a).sum()) * sqrt((b * b).sum()))).item(Float.self)
    let scale = sqrt((wantF * wantF).mean()).item(Float.self)
    // Fused class-token row only (the identity signal): cosine over positions where the mask is 1.
    let m = ctm.asType(.float32)[0..., 0..., .newAxis]                    // [1,77,1]
    let outRow = (out * m).sum(axis: 1).reshaped(-1)                      // [2048]
    let wantRow = (wantF * m).sum(axis: 1).reshaped(-1)
    let rowCos = ((outRow * wantRow).sum() / (sqrt((outRow * outRow).sum()) * sqrt((wantRow * wantRow).sum()))).item(Float.self)
    return "shape=\(out.shape) maxAbs=\(maxd) meanAbs=\(meand) cosine=\(cos) fusedRowCosine=\(rowCos) refRMS=\(scale)"
}
