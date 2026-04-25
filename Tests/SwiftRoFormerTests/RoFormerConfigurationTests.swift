import Testing
@testable import SwiftRoFormer

@Test func kimVocal2Defaults() {
    let config = RoFormerConfiguration.kimVocal2
    #expect(config.dim == 384)
    #expect(config.depth == 6)
    #expect(config.heads == 8)
    #expect(config.dimHead == 64)
    #expect(config.dimInner == 512)
    #expect(config.ffDim == 1536)
    #expect(config.freqBins == 1025)
    #expect(config.nFFT == 2048)
    #expect(config.hopLength == 441)
    #expect(config.numBands == 60)
}
