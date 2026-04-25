import MLX
import MLXNN

/// Identity placeholder module for positions in a PyTorch Sequential that have no parameters.
///
/// Used to fill array positions for Tanh, GELU, Dropout, and GLU layers
/// so that the module array aligns with PyTorch's Sequential indices.
class NoOpModule: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

/// Full mask estimator for all bands.
///
/// PyTorch structure (for `maskEstimatorDepth = d`):
/// ```
/// mask_estimators = ModuleList[MaskEstimationModule]  (1 stem)
///   to_freqs = ModuleList[                            (numBands entries)
///     Sequential(                                     (outer: [MLP, GLU])
///       Sequential(                                   (inner MLP)
///         Linear(dim → hidden),  Tanh,                 # indices 0, 1
///         Linear(hidden → hidden), Tanh,               # indices 2, 3   ┐ repeats
///         ...                                          # ...             ┘ (d−1) times
///         Linear(hidden → bandDim × 2)                 # index 2d
///       ),
///       GLU()                                         (no params)
///     )
///   ]
/// ```
///
/// With `depth = d`, the inner MLP has `(d + 1)` `Linear` layers interleaved
/// with `d` `Tanh` activations — total sequence length `2d + 1`.
/// Kim Vocal 2 / viperx / zfturbo_bs_roformer: `d = 2` → 5 entries `[L, T, L, T, L]`.
/// zfturbo_vocals_v1: `d = 1` → 3 entries `[L, T, L]`.
///
/// Weight key paths (linears at even indices only):
/// ```
/// mask_estimators.0.to_freqs.{band}.0.{0, 2, …, 2d}.weight/bias
/// ```
///
/// `toFreqs` is `[[[Module]]]`:
/// - Outer: numBands entries
/// - Middle: 1-element array for outer Sequential (only index 0 = MLP has params)
/// - Inner: `2d + 1` entries for the MLP Sequential (Tanh placeholders kept as
///   `NoOpModule` to preserve PyTorch's index scheme)
class MaskEstimator: Module {
    @ModuleInfo(key: "to_freqs") var toFreqs: [[[Module]]]

    /// Per-band output dimensions for GLU.
    let bandDims: [Int]

    /// MLP depth — number of `Tanh` nonlinearities. `depth + 1` linears total.
    let depth: Int

    init(config: RoFormerConfiguration, bandDims: [Int]) {
        precondition(config.maskEstimatorDepth >= 1,
                     "maskEstimatorDepth must be ≥ 1 (got \(config.maskEstimatorDepth))")
        self.bandDims = bandDims
        self.depth = config.maskEstimatorDepth

        let depth = config.maskEstimatorDepth
        self._toFreqs.wrappedValue = bandDims.map { bandDim -> [[Module]] in
            let dimOut = bandDim * 2  // doubled for GLU
            var inner: [Module] = []
            inner.reserveCapacity(2 * depth + 1)

            // First Linear: dim → hidden
            inner.append(Linear(config.dim, config.mlpHidden))
            inner.append(NoOpModule())  // Tanh placeholder

            // (depth − 1) hidden blocks: Linear(hidden → hidden), Tanh
            for _ in 0..<(depth - 1) {
                inner.append(Linear(config.mlpHidden, config.mlpHidden))
                inner.append(NoOpModule())
            }

            // Final Linear: hidden → dimOut (×2 for GLU)
            inner.append(Linear(config.mlpHidden, dimOut))

            // Outer Sequential: only index 0 (the MLP) carries parameters
            return [inner]
        }
    }

    /// Estimate per-band masks from transformer features.
    ///
    /// - Parameter x: Band features `[B, T, numBands, dim]`.
    /// - Returns: Concatenated per-band masks `[B, T, totalBandDim]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var bandMasks = [MLXArray]()
        bandMasks.reserveCapacity(toFreqs.count)

        for (i, bandModules) in toFreqs.enumerated() {
            let bandFeatures = x[0..., 0..., i, 0...]  // [B, T, dim]
            let inner = bandModules[0]

            // Apply linears at even indices with Tanh between.
            // Sequence length = 2*depth + 1: indices 0, 2, …, 2d are linears.
            var out = tanh((inner[0] as! Linear)(bandFeatures))
            for layerIdx in 1..<depth {
                let linearIdx = 2 * layerIdx  // 2, 4, …, 2(depth−1)
                out = tanh((inner[linearIdx] as! Linear)(out))
            }
            out = (inner[2 * depth] as! Linear)(out)  // output linear (no tanh)

            // GLU: split and gate
            let dimOut = bandDims[i]
            let first = out[0..., 0..., ..<dimOut]
            let second = out[0..., 0..., dimOut...]
            let mask = first * sigmoid(second)  // [B, T, dimOut]

            bandMasks.append(mask)
        }

        return concatenated(bandMasks, axis: -1)
    }
}
