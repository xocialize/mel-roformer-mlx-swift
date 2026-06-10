import Foundation
import MLX
import MLXNN

/// Errors that can occur during weight loading.
public enum WeightLoadingError: Error, CustomStringConvertible {
    case weightsDirectoryNotFound(String)
    case weightsFileNotFound(String)
    case keyMismatch(String)
    case loadFailed(String)

    public var description: String {
        switch self {
        case .weightsDirectoryNotFound(let path):
            return "Weights directory not found: \(path)"
        case .weightsFileNotFound(let path):
            return "Weights file not found: \(path). Run the conversion script first."
        case .keyMismatch(let detail):
            return "Weight key mismatch: \(detail)"
        case .loadFailed(let detail):
            return "Failed to load weights: \(detail)"
        }
    }
}

/// Loads safetensors weight files into MLX Module parameter trees.
public struct WeightLoader {

    /// Expected weight file name for the Kim Vocal 2 Mel-RoFormer model.
    static let vocalsWeightsFile = "mel_roformer_vocals.safetensors"

    /// Sanitize raw safetensors keys to match the Swift module tree.
    ///
    /// Partial mirror of the Python `MelRoFormer.sanitize()` in mlx-audio —
    /// applies the remappings that Swift's module tree requires:
    ///
    /// - Unwraps `to_out.0.weight` → `to_out.weight` (PyTorch wraps output
    ///   projection in `Sequential(Linear, Dropout)`; Swift uses bare Linear).
    ///
    /// Differences from the Python sanitize:
    /// - **`rotary_embed.freqs` is preserved** — Swift's `RotaryEmbedding`
    ///   module keeps the precomputed frequency buffer as a loadable
    ///   parameter, unlike the Python port which recomputes at runtime.
    /// - **`.gamma` is preserved** — Swift's `RoFormerRMSNorm` uses `gamma`
    ///   as its property name (matching PyTorch), so no rename is needed.
    /// - **QKV split is not performed here** — the `convert.py` script
    ///   splits `to_qkv.weight` → `to_q/to_k/to_v.weight` at conversion time.
    ///
    /// The Kim Vocal 2 `mel_roformer_vocals.safetensors` file shipped with
    /// this package was already fully Swift-ready. The `convert.py` output
    /// leaves `to_out.0.weight` wrapped, so this pass is what lets Swift
    /// load converted-from-PyTorch files directly.
    public static func sanitize(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        result.reserveCapacity(arrays.count)
        for (key, value) in arrays {
            var newKey = key
            if newKey.hasSuffix("to_out.0.weight") {
                newKey = String(newKey.dropLast(".0.weight".count)) + ".weight"
            }
            result[newKey] = value
        }
        return result
    }

    /// Load weights from a safetensors file into a Module.
    ///
    /// - Parameters:
    ///   - module: The MLX Module to load weights into.
    ///   - url: Path to the `.safetensors` file.
    /// - Throws: `WeightLoadingError` if the file is missing or keys don't match.
    public static func loadWeights(into module: Module, from url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw WeightLoadingError.weightsFileNotFound(url.path)
        }

        do {
            let rawArrays = try MLX.loadArrays(url: url)
            let arrays = sanitize(rawArrays)
            let parameters = ModuleParameters.unflattened(arrays)
            try module.update(parameters: parameters, verify: .noUnusedKeys)
            MLX.eval(module.parameters())
        } catch let error as WeightLoadingError {
            throw error
        } catch {
            throw WeightLoadingError.loadFailed(error.localizedDescription)
        }
    }

    /// Validate that the weights directory contains the expected files.
    ///
    /// - Parameter directory: Path to the weights directory.
    /// - Throws: `WeightLoadingError` if required files are missing.
    public static func validateWeightsDirectory(_ directory: URL) throws {
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw WeightLoadingError.weightsDirectoryNotFound(directory.path)
        }

        let filePath = directory.appendingPathComponent(vocalsWeightsFile).path
        guard FileManager.default.fileExists(atPath: filePath) else {
            throw WeightLoadingError.weightsFileNotFound(filePath)
        }
    }
}
