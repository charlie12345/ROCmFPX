FROM ubuntu:24.04 AS build

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git libcurl4-openssl-dev libvulkan-dev glslc ninja-build spirv-headers \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build -G Ninja -DGGML_VULKAN=ON \
        -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
    && cmake --build build --target llama-cli llama-server llama-bench llama-quantize -j "$(nproc)"

FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates libcurl4 libgomp1 libvulkan1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/bin/ /usr/local/bin/
ENV LD_LIBRARY_PATH=/usr/local/bin
ENV ROCMFPX_PLUGIN_PATH=/opt/rocmfpx/plugins
VOLUME ["/models", "/opt/rocmfpx/plugins", "/var/cache/rocmfpx"]
EXPOSE 8080
ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
