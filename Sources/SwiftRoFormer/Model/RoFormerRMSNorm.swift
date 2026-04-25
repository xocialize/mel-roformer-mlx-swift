import MLX
import MLXNN

/// Custom RMSNorm matching lucidrains' L2-normalize-and-scale formulation.
///
/// **NOT** equivalent to `MLXNN.RMSNorm`. lucidrains uses L2-normalization:
///
/// ```python
/// output = F.normalize(x, dim=-1) * sqrt(dim) * gamma
/// ```
///
/// This normalizes each vector to unit length, then scales by `sqrt(dim)` and
/// a learnable `gamma` parameter. Standard RMSNorm divides by `sqrt(mean(x²) + eps)`
/// which is a fundamentally different operation.
///
/// ### Epsilon convention
///
/// PyTorch `F.normalize` applies eps as `max(||x||₂, eps)` — clamping the
/// *denominator* after the sqrt, not the sum-of-squares before it. Clamping
/// inside the sqrt (e.g. `sqrt(clip(||x||₂², min: 1e-12))`) diverges from
/// F.normalize for small norms and caps the SDR ceiling against a PyTorch
/// reference. The mlx-audio Python port (SDR 58 dB vs PyTorch) matches the
/// denominator-clamp convention exactly, so this implementation mirrors it.
///
/// Weight key: `gamma` (shape `[dim]`)
class RoFormerRMSNorm: Module {

    /// Learnable scale parameter, loaded from checkpoint.
    let gamma: MLXArray

    /// Constant scale factor: `sqrt(dim)`.
    let scale: Float

    /// `F.normalize` default epsilon. Clamps the denominator after the sqrt.
    private static let eps: Float = 1e-12

    init(dim: Int) {
        self.scale = Float(dim).squareRoot()
        self.gamma = MLXArray.ones([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // L2 normalize with F.normalize eps semantic: x / max(||x||₂, eps)
        let squaredSum = sum(x * x, axis: -1, keepDims: true)
        let norm = sqrt(squaredSum)
        let normalized = x / maximum(norm, MLXArray(Self.eps))
        // Scale by sqrt(dim) and learnable gamma
        return normalized * scale * gamma
    }
}
