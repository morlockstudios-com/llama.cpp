// swift-tools-version:5.3

import PackageDescription

let package = Package(
    name: "llama",
    platforms: [.macOS(.v11),
                .iOS(.v14),
                .watchOS(.v4),
                .tvOS(.v14)
    ],
    products: [
        .library(name: "llama", targets: ["llama"]),
        .library(name: "Bert", targets: ["Bert"])
    ],
    targets: [
        .target(
            name: "llama",
            path: ".",
            sources: [
                "ggml.c",
                "llama.cpp",
                "ggml-alloc.c",
                "ggml-backend.c",
                "ggml-quants.c",
                "ggml-metal.m"
            ],
            resources: [
                .process("ggml-metal.metal")
            ],
            publicHeadersPath: "spm-headers",
            cSettings: [
                .unsafeFlags(["-Wno-shorten-64-to-32",
                              "-Ofast",
                              "-DNDEBUG"]),
                .define("GGML_USE_K_QUANTS"),
                .define("GGML_USE_ACCELERATE"),
                .define("NDEBUG"),
                .define("_XOPEN_SOURCE", to: "600"),
                .define("_DARWIN_C_SOURCE"),
                .unsafeFlags(["-fno-objc-arc"]),
                .define("GGML_SWIFT"),
                .define("GGML_USE_METAL")
            ],
            cxxSettings: [
                .unsafeFlags(["-Wno-shorten-64-to-32",
                              "-Ofast"]),
                .define("GGML_USE_K_QUANTS"),
                .define("GGML_USE_ACCELERATE"),
                .unsafeFlags(["-fno-objc-arc"]),
                .define("GGML_SWIFT"),
                .define("GGML_USE_METAL"),
                .define("NDEBUG"),
                .define("_XOPEN_SOURCE", to: "600"),
                .define("_DARWIN_C_SOURCE")
                ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit")
            ]
        ),
        .target(
            name: "Bert",
            dependencies: [ "llama" ],
            resources: [
                .process("Resources")
            ],
            publicHeadersPath: "include",
            cSettings: [
                .define("GGML_USE_ACCELERATE"),
                .unsafeFlags([
                    "-Ofast", "-DNDEBUG", "-std=gnu11"
                ])
            ],
            cxxSettings: [
                .define("GGML_USE_ACCELERATE"),
                .unsafeFlags([
                    "-Ofast", "-DNDEBUG", "-std=gnu++20"
                ])
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation"),
                .linkedFramework("NaturalLanguage")
            ]
        ),
        .testTarget(name: "BertTests",
                    dependencies: ["Bert"],
                    resources: [
                        .process("resources")
                    ]
        )
    ],
    cLanguageStandard: .c11,
    cxxLanguageStandard: .cxx11
)

