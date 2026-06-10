import Foundation
import MLX
import Testing
@testable import SwiftRoFormer

/// Pins the RoPE pairing convention to INTERLEAVED adjacent pairs (x[2i], x[2i+1]) —
/// the `rotary_embedding_torch` convention the Mel-Band-RoFormer checkpoints were
/// trained with. The "halved" convention ((x[:half], x[half:])) is a valid RoPE but
/// decorrelates attention from the trained weights: energies stay healthy while the
/// estimated masks collapse to ~0 (vocal output = digital silence). Caught live
/// (2026-06-10) and fixed via elementwise parity vs the Python mlx-audio reference.
///
/// Runs under `xcodebuild test` (MLX eval needs the staged metallib).
struct RoPEConventionTests {

    @Test func ropeRotatesAdjacentPairs() {
        // One head, dimHead 4, two base freqs f0/f1. Position t=1, basis vector e0.
        // Interleaved convention: e0 rotates within the (dim0, dim1) plane:
        //   x'[0] = cos(f0), x'[1] = sin(f0), x'[2..3] = 0
        // Halved convention would instead put the sine into dim 2 (the old bug).
        let freqs = MLXArray([Float(0.5), Float(0.25)])
        var data = [Float](repeating: 0, count: 2 * 1 * 4) // [B=1, T=2, H=1, Dh=4]
        data[4] = 1.0 // t=1, dim0 = 1
        let x = MLXArray(data, [1, 2, 1, 4])

        let out = RoFormerAttention.applyRoPE(x, freqs: freqs)
        MLX.eval(out)
        let t1 = out[0, 1, 0, 0...].asType(.float32).asArray(Float.self)

        #expect(abs(t1[0] - cos(0.5)) < 1e-5)   // cos into dim 0
        #expect(abs(t1[1] - sin(0.5)) < 1e-5)   // sin into dim 1 (ADJACENT — interleaved)
        #expect(abs(t1[2]) < 1e-6)              // nothing leaks into the second half
        #expect(abs(t1[3]) < 1e-6)
    }

    @Test func ropePreservesNorm() {
        // Rotations preserve per-vector norms regardless of convention — the property
        // that made this bug invisible to energy probes.
        let freqs = MLXArray([Float(0.7), Float(0.13)])
        let vals: [Float] = [0.3, -1.2, 0.8, 0.5, -0.6, 0.9, 1.1, -0.4]
        let x = MLXArray(vals, [1, 2, 1, 4])
        let out = RoFormerAttention.applyRoPE(x, freqs: freqs)
        MLX.eval(out)
        let inN = MLX.sum(x * x).item(Float.self)
        let outN = MLX.sum(out * out).item(Float.self)
        #expect(abs(inN - outN) < 1e-4)
    }
}
