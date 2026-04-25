import MLX
import MLXNN

/// Container for stored RoPE frequencies from the checkpoint.
///
/// Weight key: `freqs` (shape `[dimHead/2]`)
class RotaryEmbedding: Module {
    /// Precomputed frequency bases, loaded from checkpoint.
    /// Shape: `[dimHead/2]` (e.g., `[32]` for dimHead=64).
    let freqs: MLXArray

    init(dimHead: Int) {
        self.freqs = MLXArray.zeros([dimHead / 2])
    }
}

/// Multi-head attention with Rotary Position Embeddings and per-head sigmoid gates.
///
/// Weight keys (per attention block):
/// - `norm.gamma` — Pre-attention RMSNorm
/// - `rotary_embed.freqs` — Stored RoPE frequencies
/// - `to_q.weight` — Query projection `[dimInner, dim]`
/// - `to_k.weight` — Key projection `[dimInner, dim]`
/// - `to_v.weight` — Value projection `[dimInner, dim]`
/// - `to_gates.weight/bias` — Per-head sigmoid gates `[heads, dim]`
/// - `to_out.weight` — Output projection `[dim, dimInner]`
class RoFormerAttention: Module {
    @ModuleInfo var norm: RoFormerRMSNorm
    @ModuleInfo(key: "rotary_embed") var rotaryEmbed: RotaryEmbedding
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_gates") var toGates: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear

    let numHeads: Int
    let dimHead: Int
    let scale: Float

    init(dim: Int, heads: Int, dimHead: Int) {
        self.numHeads = heads
        self.dimHead = dimHead
        self.scale = 1.0 / Float(dimHead).squareRoot()

        let dimInner = heads * dimHead

        self._norm.wrappedValue = RoFormerRMSNorm(dim: dim)
        self._rotaryEmbed.wrappedValue = RotaryEmbedding(dimHead: dimHead)
        self._toQ.wrappedValue = Linear(dim, dimInner, bias: false)
        self._toK.wrappedValue = Linear(dim, dimInner, bias: false)
        self._toV.wrappedValue = Linear(dim, dimInner, bias: false)
        self._toGates.wrappedValue = Linear(dim, heads)
        self._toOut.wrappedValue = Linear(dimInner, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        let normed = norm(x)

        // Project to Q, K, V and reshape for multi-head: [B, T, H, D_h]
        var q = toQ(normed).reshaped([B, T, numHeads, dimHead])
        var k = toK(normed).reshaped([B, T, numHeads, dimHead])
        let v = toV(normed).reshaped([B, T, numHeads, dimHead])

        // Apply Rotary Position Embeddings
        q = RoFormerAttention.applyRoPE(q, freqs: rotaryEmbed.freqs)
        k = RoFormerAttention.applyRoPE(k, freqs: rotaryEmbed.freqs)

        // Transpose to [B, H, T, D_h] for attention
        let qT = q.transposed(0, 2, 1, 3)
        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        // Scaled dot-product attention
        let scores = matmul(qT * MLXArray(scale), kT.transposed(0, 1, 3, 2))
        let weights = softmax(scores, axis: -1)
        let attended = matmul(weights, vT)  // [B, H, T, D_h]

        // Per-head sigmoid gates: [B, T, H] → [B, H, T, 1]
        let gates = sigmoid(toGates(normed))  // [B, T, H]
        let gatesReshaped = gates.transposed(0, 2, 1).expandedDimensions(axis: -1)  // [B, H, T, 1]
        let gated = attended * gatesReshaped

        // Transpose back and project: [B, T, H*D_h]
        let output = gated.transposed(0, 2, 1, 3).reshaped([B, T, numHeads * dimHead])
        return toOut(output)
    }

    /// Apply Rotary Position Embeddings.
    ///
    /// - Parameters:
    ///   - x: Input tensor `[B, T, H, D_h]`.
    ///   - freqs: Frequency bases `[D_h/2]` from checkpoint.
    /// - Returns: Rotated tensor `[B, T, H, D_h]`.
    static func applyRoPE(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
        let seqLen = x.shape[1]
        let halfDim = x.shape[3] / 2

        // Build position-dependent angles: [T, halfDim]
        let positions = MLXArray(Array(0..<seqLen).map { Float($0) })
        // outer product: positions [T] × freqs [halfDim] → [T, halfDim]
        let angles = positions.reshaped([seqLen, 1]) * freqs.reshaped([1, halfDim])
        let cosAngles = cos(angles)  // [T, halfDim]
        let sinAngles = sin(angles)  // [T, halfDim]

        // Split x into paired halves along last dim
        let x1 = x[0..., 0..., 0..., ..<halfDim]   // [B, T, H, halfDim]
        let x2 = x[0..., 0..., 0..., halfDim...]    // [B, T, H, halfDim]

        // Broadcast angles to match: [1, T, 1, halfDim]
        let cos4d = cosAngles.reshaped([1, seqLen, 1, halfDim])
        let sin4d = sinAngles.reshaped([1, seqLen, 1, halfDim])

        // Apply rotation: (x1·cos - x2·sin, x2·cos + x1·sin)
        let rotated1 = x1 * cos4d - x2 * sin4d
        let rotated2 = x2 * cos4d + x1 * sin4d

        return concatenated([rotated1, rotated2], axis: -1)
    }
}
