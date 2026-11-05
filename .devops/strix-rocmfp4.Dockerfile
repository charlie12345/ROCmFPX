ARG UBUNTU_VERSION=24.04
ARG ROCM_VERSION=10.1.0
ARG BASE_ROCM_DEV_CONTAINER=rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}-full

FROM ${BASE_ROCM_DEV_CONTAINER} AS build

ARG CMAKE_HIP_ARCHITECTURES=gfx1151

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    glslc \
    libgomp1 \
    libssl-dev \
    libvulkan-dev \
    libxcb-cursor-dev \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    spirv-headers

WORKDIR /app

COPY . .

RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    GGML_HIP_ROCWMMA_FATTN=OFF \
    CMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES}" \
    scripts/build-strix-rocmfp4-mtp.sh

RUN mkdir -p /app/lib \
    && find build-strix-rocmfp4 -name "*.so*" -exec cp -P {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build-strix-rocmfp4/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

FROM ${BASE_ROCM_DEV_CONTAINER} AS base

RUN apt-get update \
    && apt-get install -y \
    curl \
    libegl1 \
    libgl1 \
    libgles2 \
    libglvnd0 \
    libglx0 \
    libgomp1 \
    libvulkan1 \
    mesa-vulkan-drivers \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

COPY --from=build /app/lib/ /app

FROM base AS full

COPY --from=build /app/full /app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-wheel \
    && pip install --break-system-packages --upgrade setuptools \
    && pip install --break-system-packages -r requirements.txt \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

ENTRYPOINT ["/app/tools.sh"]

FROM base AS light

COPY --from=build /app/full/llama-cli /app/full/llama-completion /app

WORKDIR /app

ENTRYPOINT ["/app/llama-cli"]

FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/full/llama-server /app

WORKDIR /app

HEALTHCHECK CMD ["curl", "-f", "http://localhost:8080/health"]

ENTRYPOINT ["/app/llama-server"]
