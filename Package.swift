// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "MLXImageEdit",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [ .library(name: "MLXImageEdit", targets: ["MLXImageEdit"]) ],
    dependencies: [
        // Accepts 0.31.6 through 0.32.x: the sources build against mlx-swift 0.32.3 (the floor set by
        // mlx-swift-lm 3.31.4 in the esh graph). An explicit range avoids needing a SwiftPM path override
        // just to clear an over-tight upper bound.
        .package(url: "https://github.com/ml-explore/mlx-swift", "0.31.6" ..< "0.33.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMajor(from: "1.3.0")),
    ],
    targets: [
        .target(name: "MLXImageEdit", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXRandom", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift"),
            .product(name: "Hub", package: "swift-transformers"),
        ]),
        .executableTarget(name: "ip2p-validate", dependencies: ["MLXImageEdit"]),
        .testTarget(name: "MLXImageEditTests", dependencies: ["MLXImageEdit"]),
    ]
)
