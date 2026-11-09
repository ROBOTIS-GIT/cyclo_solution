# Jetson TensorRT dependency

The robot uses L4T 36.4.3. Its AI Worker base image has a zero-byte `libnvdla_compiler.so` placeholder and the host does not supply the library. TensorRT links it even for GPU-only inference. The Dockerfile extracts just the library into `/opt/demo-trt/lib` from NVIDIA's matching package, without changing host packages or drivers.

Source: https://repo.download.nvidia.com/jetson/common/pool/main/n/nvidia-l4t-dla-compiler/nvidia-l4t-dla-compiler_36.4.3-20250107174145_arm64.deb

SHA256: `928170c9e743257c063c235a610473cfec4be917a36f91a61a890ba2bfc47c94`

The package contains NVIDIA's copyright/license information; it is not project code. This workaround is specific to the tested Orin/L4T 36.4.3 base. Other JetPack versions require their matching package. AMD64 builds skip extraction.
