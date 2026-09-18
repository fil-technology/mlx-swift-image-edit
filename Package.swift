// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "MLXImageEdit",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [ .library(name: "MLXImageEdit", targets: ["MLXImageEdit"]) ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.29.1")),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.0.0")),
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
