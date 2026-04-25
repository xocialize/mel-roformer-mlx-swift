import Foundation
import MLX
import MLXNN
import SwiftRoFormer

// MARK: - Helpers

func log(_ message: String) {
    print("[\(timestamp())] \(message)")
}

func timestamp() -> String {
    let f = DateFormatter()
    f.dateFormat = "HH:mm:ss.SSS"
    return f.string(from: Date())
}

func pass(_ name: String) {
    print("  \u{2705} PASS: \(name)")
}

func fail(_ name: String, _ detail: String) {
    print("  \u{274C} FAIL: \(name) — \(detail)")
}

// MARK: - Weight Path Resolution

let weightsDir: URL = {
    // Check command-line argument first
    if CommandLine.arguments.count > 1 {
        return URL(fileURLWithPath: CommandLine.arguments[1])
    }
    // Default: weights/ directory relative to package root
    let binary = URL(fileURLWithPath: CommandLine.arguments[0])
    // Walk up from .build/arm64-apple-macosx/debug/roformer-validate
    let packageRoot = binary
        .deletingLastPathComponent()  // debug/
        .deletingLastPathComponent()  // arm64-apple-macosx/
        .deletingLastPathComponent()  // .build/
        .deletingLastPathComponent()  // package root
    return packageRoot.appendingPathComponent("weights")
}()

let weightsFile = weightsDir.appendingPathComponent("mel_roformer_vocals.safetensors")

// MARK: - Tests

log("RoFormer Validation Suite")
log("Weights: \(weightsFile.path)")
print()

var passed = 0
var failed = 0
var skipped = 0

guard FileManager.default.fileExists(atPath: weightsFile.path) else {
    log("Weights file not found at \(weightsFile.path)")
    log("Provide path: roformer-validate <weights-directory>")
    exit(1)
}

// Test 1: Safetensors key count
do {
    log("Test 1: Safetensors file contains exactly 708 keys")
    let arrays = try MLX.loadArrays(url: weightsFile)
    if arrays.count == 708 {
        pass("Key count = \(arrays.count)")
        passed += 1
    } else {
        fail("Key count", "Expected 708, got \(arrays.count)")
        failed += 1
    }
} catch {
    fail("Key count", "Load error: \(error)")
    failed += 1
}

// Test 2: Model instantiation
log("Test 2: Model instantiation")
let model = MelRoFormer()
pass("MelRoFormer() created")
passed += 1

// Test 3: Weight loading with verification
do {
    log("Test 3: Load all 708 weights (strict verification)")
    try WeightLoader.loadWeights(into: model, from: weightsFile)
    pass("All weights loaded — module tree matches checkpoint")
    passed += 1
} catch {
    fail("Weight loading", "\(error)")
    failed += 1
    // If weight loading fails, remaining tests are meaningless
    log("\nWeight loading failed — skipping forward pass tests.")
    log("\nResults: \(passed) passed, \(failed) failed")
    exit(1)
}

// Test 4: Forward pass shape
do {
    log("Test 4: Forward pass produces correct output shape")
    let samples = 44100  // 1 second at 44.1kHz
    let input = MLXArray.zeros([1, 2, samples])

    let output = model(input)
    MLX.eval(output)

    let expectedShape = [1, 2, samples]
    if output.shape == expectedShape {
        pass("Output shape: \(output.shape)")
        passed += 1
    } else {
        fail("Output shape", "Expected \(expectedShape), got \(output.shape)")
        failed += 1
    }
} catch {
    fail("Forward pass shape", "\(error)")
    failed += 1
}

// Test 5: Forward pass no NaN
do {
    log("Test 5: Forward pass produces no NaN values")
    let samples = 22050  // 0.5 seconds
    let input = MLXRandom.normal([1, 2, samples]) * 0.1

    let output = model(input)
    MLX.eval(output)

    let hasNaN = MLX.any(MLX.isNaN(output)).item(Bool.self)
    if !hasNaN {
        pass("No NaN values in output")
        passed += 1
    } else {
        fail("NaN check", "Output contains NaN values")
        failed += 1
    }
} catch {
    fail("NaN check", "\(error)")
    failed += 1
}

// Test 6: Forward pass energy check (output should not be all zeros)
do {
    log("Test 6: Forward pass produces non-trivial output")
    let samples = 22050
    let input = MLXRandom.normal([1, 2, samples]) * 0.1

    let output = model(input)
    MLX.eval(output)

    let energy = MLX.sum(MLX.multiply(output, output)).item(Float.self)
    if energy > 1e-10 {
        pass("Output energy: \(energy) (non-trivial)")
        passed += 1
    } else {
        fail("Energy check", "Output energy \(energy) is effectively zero")
        failed += 1
    }
} catch {
    fail("Energy check", "\(error)")
    failed += 1
}

// Test 7: End-to-end separation on real audio
do {
    log("Test 7: End-to-end vocal separation on test audio")

    // Locate test fixture
    let binary = URL(fileURLWithPath: CommandLine.arguments[0])
    let packageRoot = binary
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let testAudio = packageRoot
        .appendingPathComponent("tests/fixtures/audio/synthetic_mix.wav")

    guard FileManager.default.fileExists(atPath: testAudio.path) else {
        log("  ⏭ SKIP: Test audio not found at \(testAudio.path)")
        skipped += 1
        // Jump to summary
        print()
        log("Results: \(passed) passed, \(failed) failed, \(skipped) skipped")
        exit(failed > 0 ? 1 : 0)
    }

    log("  Audio: \(testAudio.lastPathComponent)")

    // Create separator with already-loaded weights
    let separator = try await RoFormerSeparator(weightsDirectory: weightsDir)

    // Create temp output directory
    let outputDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("roformer-validate-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: outputDir) }

    let startTime = CFAbsoluteTimeGetCurrent()

    // Run separation (explicit type annotation to select async overload)
    let result: StemResult = try await separator.separateVocals(from: testAudio, to: outputDir)
    let elapsed = CFAbsoluteTimeGetCurrent() - startTime

    // Verify outputs exist and have content
    let vocalsExists = FileManager.default.fileExists(atPath: result.vocalsURL.path)
    let accompExists = FileManager.default.fileExists(atPath: result.accompanimentURL.path)
    let vocalsSize = (try? FileManager.default.attributesOfItem(atPath: result.vocalsURL.path)[.size] as? Int) ?? 0
    let accompSize = (try? FileManager.default.attributesOfItem(atPath: result.accompanimentURL.path)[.size] as? Int) ?? 0

    if vocalsExists && accompExists && vocalsSize > 1000 && accompSize > 1000 {
        pass("Separation complete in \(String(format: "%.1f", elapsed))s")
        log("    Vocals: \(vocalsSize / 1024) KB at \(result.vocalsURL.lastPathComponent)")
        log("    Accompaniment: \(accompSize / 1024) KB at \(result.accompanimentURL.lastPathComponent)")
        log("    Duration: \(String(format: "%.1f", result.durationSeconds))s audio")
        log("    Realtime factor: \(String(format: "%.1f", result.realtimeFactor))x")
        passed += 1
    } else {
        fail("Separation output", "Vocals exists=\(vocalsExists) (\(vocalsSize)B), Accompaniment exists=\(accompExists) (\(accompSize)B)")
        failed += 1
    }
} catch {
    fail("End-to-end separation", "\(error)")
    failed += 1
}

// Summary
print()
log("Results: \(passed) passed, \(failed) failed, \(skipped) skipped")
exit(failed > 0 ? 1 : 0)
