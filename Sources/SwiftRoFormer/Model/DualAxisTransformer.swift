import MLX
import MLXNN

/// Single-axis transformer with output norm.
///
/// Matches PyTorch `Transformer(depth=1)` which wraps its blocks in:
/// ```
/// layers = ModuleList[ModuleList[Attention, FeedForward]] × depth
/// norm = RMSNorm(dim)
/// ```
///
/// For Kim Vocal 2 (depth=1), this produces:
/// - `layers.0.0.*` — Attention
/// - `layers.0.1.*` — FeedForward
/// - `norm.gamma` — Output RMSNorm
///
/// The `layers` property is `[[Module]]` — an array of depth levels,
/// each containing `[RoFormerAttention, RoFormerFFN]`.
/// This matches PyTorch's `ModuleList[ModuleList[...]]` exactly.
class Transformer: Module {
    @ModuleInfo var layers: [[Module]]
    @ModuleInfo var norm: RoFormerRMSNorm

    init(dim: Int, depth: Int, heads: Int, dimHead: Int, ffMult: Int) {
        // depth=1: `layers.0` = [Attention, FFN]
        self._layers.wrappedValue = (0..<depth).map { _ -> [Module] in
            [
                RoFormerAttention(dim: dim, heads: heads, dimHead: dimHead),
                RoFormerFFN(dim: dim, ffMult: ffMult),
            ]
        }
        self._norm.wrappedValue = RoFormerRMSNorm(dim: dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for pair in layers {
            let attention = pair[0] as! RoFormerAttention
            let ffn = pair[1] as! RoFormerFFN
            out = out + attention(out)
            out = out + ffn(out)
        }
        return norm(out)
    }
}
