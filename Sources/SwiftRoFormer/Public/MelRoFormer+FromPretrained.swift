// MelRoFormer+FromPretrained.swift
//
// HuggingFace Hub convenience for downloading a published Mel-Band-RoFormer
// MLX checkpoint and constructing a ready-to-run model in one call.
//
// The published mlx-community repos pair `model.safetensors` with a
// `config.json` whose `checkpoint_family` field names the matching preset
// (`kim_vocal_2`, `zfturbo_vocals_v1`, etc.). This extension reads that field
// to dispatch the right `RoFormerConfiguration` automatically — same
// ergonomics as the Python `MelRoFormer.from_pretrained()` helper.

import Foundation
import Hub

/// Errors specific to Hub-based loading.
public enum FromPretrainedError: Error, CustomStringConvertible {
    case configFileMissing(URL)
    case weightsFileMissing(URL)
    case configParseFailed(String)
    case unknownCheckpointFamily(String)

    public var description: String {
        switch self {
        case .configFileMissing(let url):
            return "config.json not found in downloaded snapshot at \(url.path)"
        case .weightsFileMissing(let url):
            return "model.safetensors not found in downloaded snapshot at \(url.path)"
        case .configParseFailed(let detail):
            return "Failed to parse config.json: \(detail)"
        case .unknownCheckpointFamily(let name):
            return "Unknown checkpoint_family '\(name)' in config.json. Supported: kim_vocal_2, zfturbo_vocals_v1. For other families, instantiate `MelRoFormer(config:)` directly with a matching `RoFormerConfiguration`."
        }
    }
}

extension MelRoFormer {
    /// Download a published `mlx-community` Mel-Band-RoFormer checkpoint and
    /// return a model with weights loaded.
    ///
    /// Reads the snapshot's `config.json` to dispatch the correct preset based
    /// on the `checkpoint_family` field. To override (e.g. to use a custom
    /// preset for a community-trained checkpoint), pass `configuration:`
    /// explicitly.
    ///
    /// - Parameters:
    ///   - repoId: The HuggingFace repo identifier, e.g.
    ///       `"mlx-community/mel-roformer-kim-vocal-2-mlx"`.
    ///   - configuration: Optional explicit configuration. If `nil`, the
    ///       configuration is resolved from the snapshot's `config.json`.
    ///   - hub: Optional pre-configured `HubApi` instance (e.g. with a custom
    ///       download base or auth token). Defaults to `HubApi()`.
    ///   - progress: Optional progress callback (0.0–1.0).
    ///
    /// - Returns: A `MelRoFormer` with weights loaded from the snapshot.
    ///
    /// - Throws: `FromPretrainedError` for snapshot/config issues,
    ///   `WeightLoadingError` for weight-load issues, or any error
    ///   surfaced by `HubApi`.
    ///
    /// ## Example
    ///
    /// ```swift
    /// let model = try await MelRoFormer.fromPretrained(
    ///     "mlx-community/mel-roformer-kim-vocal-2-mlx"
    /// )
    /// let vocals = model(audio)
    /// ```
    public static func fromPretrained(
        _ repoId: String,
        configuration: RoFormerConfiguration? = nil,
        hub: HubApi = HubApi(),
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> MelRoFormer {
        let repo = Hub.Repo(id: repoId)
        let snapshotURL = try await hub.snapshot(
            from: repo,
            matching: ["model.safetensors", "config.json"]
        ) { progressUpdate in
            progress?(progressUpdate.fractionCompleted)
        }

        let configURL = snapshotURL.appendingPathComponent("config.json")
        let weightsURL = snapshotURL.appendingPathComponent("model.safetensors")

        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw FromPretrainedError.configFileMissing(snapshotURL)
        }
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw FromPretrainedError.weightsFileMissing(snapshotURL)
        }

        let resolvedConfig: RoFormerConfiguration
        if let configuration {
            resolvedConfig = configuration
        } else {
            resolvedConfig = try Self.resolveConfiguration(fromConfigJSONAt: configURL)
        }

        let model = MelRoFormer(config: resolvedConfig)
        try WeightLoader.loadWeights(into: model, from: weightsURL)
        return model
    }

    /// Read `config.json` and map its `checkpoint_family` field to a preset.
    ///
    /// Internal entry point for `fromPretrained`; exposed so test code can
    /// validate the dispatch table without performing a Hub download.
    static func resolveConfiguration(fromConfigJSONAt url: URL) throws -> RoFormerConfiguration {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw FromPretrainedError.configParseFailed("could not read \(url.path): \(error.localizedDescription)")
        }

        struct Manifest: Decodable {
            let checkpoint_family: String?
        }

        let family: String
        do {
            family = try JSONDecoder().decode(Manifest.self, from: data).checkpoint_family ?? ""
        } catch {
            throw FromPretrainedError.configParseFailed(error.localizedDescription)
        }

        switch family {
        case "kim_vocal_2":
            return .kimVocal2
        case "zfturbo_vocals_v1":
            return .zfturboVocalsV1
        case "":
            throw FromPretrainedError.configParseFailed(
                "config.json is missing a `checkpoint_family` field. Pass `configuration:` explicitly to override."
            )
        default:
            throw FromPretrainedError.unknownCheckpointFamily(family)
        }
    }
}
