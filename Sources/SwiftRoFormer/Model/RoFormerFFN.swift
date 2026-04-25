import MLX
import MLXNN

/// Standard feed-forward network (NOT GLU-gated).
///
/// `RMSNorm → Linear(dim→ffDim) → GELU → Linear(ffDim→dim)`
///
/// Matches PyTorch `FeedForward` which wraps an `nn.Sequential` inside `net`:
/// ```
/// net = Sequential(RMSNorm, Linear, GELU, Dropout, Linear, Dropout)
///                    ^0       ^1     ^2     ^3      ^4      ^5
/// ```
///
/// Only indices 0, 1, 4 have learnable parameters.
/// The `net` property stores a 5-element `[Module]` array to match the Sequential.
///
/// Weight key prefix: `net.{0,1,4}.*`
class RoFormerFFN: Module {
    @ModuleInfo var net: [Module]

    init(dim: Int, ffMult: Int = 4) {
        let ffDim = dim * ffMult
        self._net.wrappedValue = [
            RoFormerRMSNorm(dim: dim),  // 0: norm
            Linear(dim, ffDim),          // 1: expand
            NoOpModule(),                // 2: GELU placeholder
            NoOpModule(),                // 3: Dropout placeholder
            Linear(ffDim, dim),          // 4: compress
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let norm = net[0] as! RoFormerRMSNorm
        let expand = net[1] as! Linear
        let compress = net[4] as! Linear
        var out = norm(x)
        out = gelu(expand(out))
        out = compress(out)
        return out
    }
}
