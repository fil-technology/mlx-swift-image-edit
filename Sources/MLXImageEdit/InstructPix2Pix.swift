import Foundation
import MLX
import MLXRandom
import Hub

// Real InstructPix2Pix on top of the vendored SD1.5 pipeline (same-module internal access to unet/vae/
// textEncoder/sampler). IP2P = SD UNet with in_channels=8 (noisy latent 4ch ++ source latent 4ch on the
// channel axis) and dual classifier-free guidance across three branches (text, image, uncond). This is NOT
// img2img: the source latent is concatenated CLEAN at every step (not merely noised as the start), which is
// what preserves subject/scene.

/// The `.instructPix2Pix` model configuration (8 files, SD1.5/CLIP-L layout). `id` need not match a hub repo
/// because we load from a local directory staged into the HubApi cache (see NativeImageEditor).
func instructPix2PixConfiguration() -> StableDiffusionConfiguration {
    StableDiffusionConfiguration(
        id: "timbrooks/instruct-pix2pix",
        files: [
            .unetConfig: "unet/config.json",
            .unetWeights: "unet/diffusion_pytorch_model.fp16.safetensors",
            .textEncoderConfig: "text_encoder/config.json",
            .textEncoderWeights: "text_encoder/model.fp16.safetensors",
            .vaeConfig: "vae/config.json",
            .vaeWeights: "vae/diffusion_pytorch_model.fp16.safetensors",
            .diffusionConfig: "scheduler/scheduler_config.json",
            .tokenizerVocabulary: "tokenizer/vocab.json",
            .tokenizerMerges: "tokenizer/merges.txt",
        ],
        defaultParameters: { EvaluateParameters(cfgWeight: 7.5, steps: 20) },
        factory: { hub, sdConfiguration, loadConfiguration in
            try InstructPix2Pix(hub: hub, configuration: sdConfiguration, dType: loadConfiguration.dType)
        }
    )
}

/// Pure, weight-free numerics for InstructPix2Pix, exposed for deterministic tests/self-test.
public enum IP2PMath {
    /// 8-channel UNet input for one branch: [noisy latent (4ch) ++ conditioning latent (4ch)] on the channel
    /// axis. `condition` is the clean source latent for the text/image branches, zeros for the uncond branch.
    public static func channelConcat(_ noisy: MLXArray, _ condition: MLXArray) -> MLXArray {
        concatenated([noisy, condition], axis: -1)
    }
    /// Dual classifier-free guidance (Brooks et al. 2023):
    /// eps = uncond + s_txt·(text − image) + s_img·(image − uncond).
    public static func combineGuidance(text: MLXArray, image: MLXArray, uncond: MLXArray,
                                       guidanceScale: Float, imageGuidanceScale: Float) -> MLXArray {
        uncond + guidanceScale * (text - image) + imageGuidanceScale * (image - uncond)
    }
}

/// The InstructPix2Pix model file layout the editor expects staged under the hub base (public for tests).
public let instructPix2PixModelFiles: [String] = [
    "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
    "scheduler/scheduler_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt",
]

final class InstructPix2Pix: StableDiffusion {
    // Ancestral Euler matches diffusers' IP2P default scheduler better than plain Euler.
    override init(
        hub: HubApi, configuration: StableDiffusionConfiguration, dType: DType,
        diffusionConfiguration: DiffusionConfiguration? = nil, unet: UNetModel? = nil,
        textEncoder: CLIPTextModel? = nil, autoencoder: Autoencoder? = nil,
        sampler: SimpleEulerSampler? = nil, tokenizer: CLIPTokenizer? = nil
    ) throws {
        let dc = try diffusionConfiguration
            ?? loadDiffusionConfiguration(hub: hub, configuration: configuration)
        try super.init(
            hub: hub, configuration: configuration, dType: dType,
            diffusionConfiguration: dc,
            sampler: sampler ?? SimpleEulerAncestralSampler(configuration: dc))
    }

    /// CLIP-L conditioning (SD1.5 uses the last hidden state). Prompt + negative are tokenized TOGETHER so
    /// both pad to a common sequence length (required for the branch concatenation on axis 0).
    private func promptAndUncond(_ prompt: String, _ negative: String) -> (text: MLXArray, uncond: MLXArray) {
        let tokens = tokenize(tokenizer: tokenizer, text: prompt, negativeText: negative)  // [2, L]
        let emb = textEncoder(tokens).lastHiddenState                                       // [2, L, 768]
        return (emb[0][.newAxis], emb[1][.newAxis])                                         // [1, L, 768] each
    }

    struct EditParameters {
        var prompt: String
        var negativePrompt: String
        var steps: Int
        var guidanceScale: Float        // text: how strongly to follow the instruction
        var imageGuidanceScale: Float   // image: how strongly to preserve the source
        var seed: UInt64
    }

    /// Run IP2P denoising. `source` is [H, W, 3] float in [-1, 1]. Calls `onStep(i,total)` each step and
    /// checks `isCancelled()` cooperatively. Returns the final latent (decode with `decode(xt:)`).
    func editLatents(source: MLXArray, params: EditParameters,
                     onStep: (Int, Int) -> Void, isCancelled: () -> Bool) throws -> MLXArray {
        MLXRandom.seed(params.seed)

        // Source latent (clean), concatenated every step. VAE.encode returns mean * scalingFactor; IP2P
        // concatenates the raw VAE mode, so divide the scaling back out.
        let (encMean, _) = autoencoder.encode(source[.newAxis])          // [1, h/8, w/8, 4]
        let imageLatents = encMean / autoencoder.scalingFactor
        let zeroImage = MLXArray.zeros(like: imageLatents)

        // Conditioning: [text, uncond, uncond] for the three branches (text, image, uncond).
        let (textEmb, uncondEmb) = promptAndUncond(params.prompt, params.negativePrompt)
        let conditioning = concatenated([textEmb, uncondEmb, uncondEmb], axis: 0)

        var xt = sampler.samplePrior(
            shape: [1] + [source.dim(0) / 8, source.dim(1) / 8] + [autoencoder.latentChannels],
            dType: dType)

        let steps = sampler.timeSteps(steps: params.steps, start: sampler.maxTime, dType: dType)
        let total = steps.count
        for (idx, (t, tPrev)) in steps.enumerated() {
            if isCancelled() { throw CancellationError() }
            // 8-channel input per branch: [noisy latent ++ conditioning latent] on the channel axis.
            let inText = IP2PMath.channelConcat(xt, imageLatents)
            let inImage = IP2PMath.channelConcat(xt, imageLatents)
            let inUncond = IP2PMath.channelConcat(xt, zeroImage)
            let modelInput = concatenated([inText, inImage, inUncond], axis: 0)   // [3, h, w, 8]
            let tUnet = broadcast(t, to: [modelInput.dim(0)])

            let eps = unet(modelInput, timestep: tUnet, encoderX: conditioning, textTime: nil)
            let parts = eps.split(parts: 3, axis: 0)
            let noisePred = IP2PMath.combineGuidance(
                text: parts[0], image: parts[1], uncond: parts[2],
                guidanceScale: params.guidanceScale, imageGuidanceScale: params.imageGuidanceScale)

            xt = sampler.step(epsPred: noisePred, xt: xt, t: t, tPrev: tPrev)
            eval(xt)
            onStep(idx + 1, total)
        }
        return xt
    }
}
