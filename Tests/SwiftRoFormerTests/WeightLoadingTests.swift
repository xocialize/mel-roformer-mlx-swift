import Testing
import Foundation
import MLX
import MLXNN
import MLXRandom
@testable import SwiftRoFormer

/// Weight-loading validation tests.
///
/// These tests validate that the Swift module tree produces key paths matching
/// the safetensors keys from the published checkpoints on
/// `mlx-community`. They require the corresponding `.safetensors` file to be
/// present at `<package-root>/weights/<name>.safetensors`.
///
/// Tests skip cleanly (via the `.enabled(if:)` trait) when weights aren't
/// present, so `swift test` runs green out-of-the-box on a fresh clone. To
/// enable, download the relevant checkpoint:
///
///     hf download mlx-community/mel-roformer-kim-vocal-2-mlx \
///         --include "model.safetensors" \
///         --local-dir weights/
///     mv weights/model.safetensors weights/mel_roformer_vocals.safetensors
///
///     hf download mlx-community/mel-roformer-zfturbo-vocals-v1-mlx \
///         --include "model.safetensors" \
///         --local-dir weights/
///     mv weights/model.safetensors weights/mel_roformer_zfturbo_v1.safetensors
struct WeightLoadingTests {

    /// Path to the weights directory (relative to package root).
    static let weightsDirectory: URL = {
        let testBundle = URL(fileURLWithPath: #filePath)
        // #filePath → Tests/SwiftRoFormerTests/WeightLoadingTests.swift
        // Package root is 3 levels up.
        let packageRoot = testBundle
            .deletingLastPathComponent()  // SwiftRoFormerTests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // package root
        return packageRoot.appendingPathComponent("weights")
    }()

    static let kimWeightsFile = weightsDirectory.appendingPathComponent("mel_roformer_vocals.safetensors")
    static let zfturboWeightsFile = weightsDirectory.appendingPathComponent("mel_roformer_zfturbo_v1.safetensors")

    static let kimWeightsAvailable = FileManager.default.fileExists(atPath: kimWeightsFile.path)
    static let zfturboWeightsAvailable = FileManager.default.fileExists(atPath: zfturboWeightsFile.path)

    // MARK: - Kim Vocal 2

    @Test("Load all 708 weights with .noUnusedKeys verification",
          .enabled(if: kimWeightsAvailable, "Kim Vocal 2 weights not present in weights/"))
    func loadWeightsWithVerification() throws {
        let model = MelRoFormer()
        // This is THE critical test — if this succeeds, all 708 key paths match.
        try WeightLoader.loadWeights(into: model, from: Self.kimWeightsFile)
    }

    @Test("Safetensors file contains exactly 708 keys",
          .enabled(if: kimWeightsAvailable, "Kim Vocal 2 weights not present in weights/"))
    func safetensorsKeyCount() throws {
        let arrays = try MLX.loadArrays(url: Self.kimWeightsFile)
        #expect(arrays.count == 708, "Expected 708 safetensors keys, got \(arrays.count)")
    }

    @Test("Forward pass produces expected output shape",
          .enabled(if: kimWeightsAvailable, "Kim Vocal 2 weights not present in weights/"))
    func forwardPassShape() throws {
        let model = MelRoFormer()
        try WeightLoader.loadWeights(into: model, from: Self.kimWeightsFile)

        // 1 second of stereo silence at 44.1kHz.
        let samples = 44100
        let input = MLXArray.zeros([1, 2, samples])

        let output = model(input)
        MLX.eval(output)

        #expect(output.shape == [1, 2, samples],
                "Expected output shape [1, 2, \(samples)], got \(output.shape)")
    }

    @Test("Forward pass produces no NaN values",
          .enabled(if: kimWeightsAvailable, "Kim Vocal 2 weights not present in weights/"))
    func forwardPassNoNaN() throws {
        let model = MelRoFormer()
        try WeightLoader.loadWeights(into: model, from: Self.kimWeightsFile)

        // 0.5s of random noise.
        let samples = 22050
        let input = MLXRandom.normal([1, 2, samples]) * 0.1

        let output = model(input)
        MLX.eval(output)

        let hasNaN = MLX.any(MLX.isNaN(output)).item(Bool.self)
        #expect(!hasNaN, "Output contains NaN values")
    }

    // MARK: - ZFTurbo vocals_v1
    //
    // Exercises three things the Kim Vocal 2 file does not:
    // 1. `maskEstimatorDepth = 1` — the 3-entry [L, T, L] inner MLP
    // 2. `dim = 192`, `depth = 8`, `hopLength = 512` — smaller/different shape
    // 3. `to_out.0.weight` keys needing the WeightLoader.sanitize unwrap

    @Test("zfturbo_vocals_v1 loads with .noUnusedKeys verification",
          .enabled(if: zfturboWeightsAvailable, "ZFTurbo weights not present in weights/"))
    func zfturboLoadWeights() throws {
        let model = MelRoFormer(config: .zfturboVocalsV1)
        try WeightLoader.loadWeights(into: model, from: Self.zfturboWeightsFile)
    }

    @Test("zfturbo_vocals_v1 forward pass produces expected shape",
          .enabled(if: zfturboWeightsAvailable, "ZFTurbo weights not present in weights/"))
    func zfturboForwardShape() throws {
        let model = MelRoFormer(config: .zfturboVocalsV1)
        try WeightLoader.loadWeights(into: model, from: Self.zfturboWeightsFile)

        let samples = 44100
        let input = MLXArray.zeros([1, 2, samples])
        let output = model(input)
        MLX.eval(output)

        #expect(output.shape == [1, 2, samples],
                "Expected output shape [1, 2, \(samples)], got \(output.shape)")
    }

    @Test("zfturbo_vocals_v1 forward pass produces no NaN values",
          .enabled(if: zfturboWeightsAvailable, "ZFTurbo weights not present in weights/"))
    func zfturboForwardNoNaN() throws {
        let model = MelRoFormer(config: .zfturboVocalsV1)
        try WeightLoader.loadWeights(into: model, from: Self.zfturboWeightsFile)

        let samples = 22050
        let input = MLXRandom.normal([1, 2, samples]) * 0.1
        let output = model(input)
        MLX.eval(output)

        let hasNaN = MLX.any(MLX.isNaN(output)).item(Bool.self)
        #expect(!hasNaN, "Output contains NaN values")
    }
}
