// Tests for MelRoFormer.fromPretrained's config-dispatch logic.
//
// These tests do NOT perform Hub downloads — they exercise the
// `resolveConfiguration(fromConfigJSONAt:)` helper directly with synthetic
// config.json files, since the Hub side is covered by integration testing
// and would require network access to run.

import Foundation
import Testing
@testable import SwiftRoFormer

private func writeTempConfig(_ contents: String) throws -> URL {
    let tmpDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("mel-roformer-test-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    let url = tmpDir.appendingPathComponent("config.json")
    try contents.write(to: url, atomically: true, encoding: .utf8)
    return url
}

@Test func resolveKimVocal2Family() throws {
    let url = try writeTempConfig(#"{"checkpoint_family": "kim_vocal_2"}"#)
    let config = try MelRoFormer.resolveConfiguration(fromConfigJSONAt: url)
    #expect(config.dim == 384)
    #expect(config.depth == 6)
    #expect(config.hopLength == 441)
}

@Test func resolveZFTurboVocalsV1Family() throws {
    let url = try writeTempConfig(#"{"checkpoint_family": "zfturbo_vocals_v1"}"#)
    let config = try MelRoFormer.resolveConfiguration(fromConfigJSONAt: url)
    #expect(config.dim == 192)
    #expect(config.depth == 8)
    #expect(config.hopLength == 512)
}

@Test func resolveAcceptsExtraConfigFields() throws {
    // Real published config.json carries _dtype, _source_sha256, _mlx_version,
    // etc. — make sure the resolver tolerates them.
    let url = try writeTempConfig(#"""
    {
      "_class": "MelRoFormerConfig",
      "model_type": "mel_band_roformer",
      "checkpoint_family": "kim_vocal_2",
      "dim": 384,
      "depth": 6,
      "_dtype": "bfloat16",
      "_mlx_version": "0.31.0"
    }
    """#)
    let config = try MelRoFormer.resolveConfiguration(fromConfigJSONAt: url)
    #expect(config.dim == 384)
    #expect(config.depth == 6)
}

@Test func resolveRejectsUnknownFamily() throws {
    let url = try writeTempConfig(#"{"checkpoint_family": "viperx_vocals"}"#)
    #expect(throws: FromPretrainedError.self) {
        _ = try MelRoFormer.resolveConfiguration(fromConfigJSONAt: url)
    }
}

@Test func resolveRejectsMissingFamily() throws {
    let url = try writeTempConfig(#"{"dim": 384, "depth": 6}"#)
    #expect(throws: FromPretrainedError.self) {
        _ = try MelRoFormer.resolveConfiguration(fromConfigJSONAt: url)
    }
}
