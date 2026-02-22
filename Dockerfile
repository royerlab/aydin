# ==============================================================================
# Aydin Docker Images
#
# Multi-stage, multi-target Dockerfile producing three image variants:
#
#   aydin         — CLI, CPU-only (default)
#   aydin-gpu     — CLI, NVIDIA CUDA GPU support
#   aydin-studio  — Full GUI (Aydin Studio) accessible via web browser
#
# Build examples (--platform required on Apple Silicon):
#   docker build --platform linux/amd64 --target aydin        -t aydin .
#   docker build --platform linux/amd64 --target aydin-gpu    -t aydin-gpu .
#   docker build --platform linux/amd64 --target aydin-studio -t aydin-studio .
#
# Or use the Makefile (handles platform automatically):
#   make docker-build
#
# See docker/README.md for full usage documentation.
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage: base — shared Python environment with all Aydin dependencies
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS base

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# System libraries required by Qt6, OpenGL, and scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        # OpenGL / rendering
        libgl1-mesa-glx \
        libegl1 \
        libglib2.0-0 \
        # Qt6 / XCB platform plugin
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-keysyms1 \
        libxcb-shape0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxkbcommon-x11-0 \
        libxkbcommon0 \
        libfontconfig1 \
        libdbus-1-3 \
        # OpenMP (LightGBM, numba)
        libomp-dev \
        # Image codec libraries
        libtiff6 \
        libjpeg62-turbo \
        libpng16-16 \
        libwebp7 \
    && rm -rf /var/lib/apt/lists/*

# Install Aydin from the local source tree
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && rm -rf /app/src /app/pyproject.toml /app/README.md

# Default working directory for user data
WORKDIR /data

# ------------------------------------------------------------------------------
# Target: aydin — CLI, CPU-only (default target)
# ------------------------------------------------------------------------------
FROM base AS aydin

LABEL org.opencontainers.image.title="Aydin" \
      org.opencontainers.image.description="Self-supervised image denoising — CLI" \
      org.opencontainers.image.url="https://github.com/royerlab/aydin" \
      org.opencontainers.image.source="https://github.com/royerlab/aydin"

ENTRYPOINT ["aydin"]
CMD ["--help"]

# ------------------------------------------------------------------------------
# Target: aydin-gpu — CLI with NVIDIA CUDA support
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS aydin-gpu

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 from deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        # System libraries (same as base)
        libgl1-mesa-glx \
        libegl1 \
        libglib2.0-0 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-keysyms1 \
        libxcb-shape0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxkbcommon-x11-0 \
        libxkbcommon0 \
        libfontconfig1 \
        libdbus-1-3 \
        libomp-dev \
        libtiff5 \
        libjpeg8 \
        libpng16-16 \
        libwebp7 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default and bootstrap pip for it
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python3 -m ensurepip --upgrade \
    && python3 -m pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support first, then Aydin
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python3 -m pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cu124 \
    && python3 -m pip install --no-cache-dir . \
    && rm -rf /app/src /app/pyproject.toml /app/README.md

WORKDIR /data

LABEL org.opencontainers.image.title="Aydin (GPU)" \
      org.opencontainers.image.description="Self-supervised image denoising — CLI with NVIDIA CUDA" \
      org.opencontainers.image.url="https://github.com/royerlab/aydin" \
      org.opencontainers.image.source="https://github.com/royerlab/aydin"

ENTRYPOINT ["aydin"]
CMD ["--help"]

# ------------------------------------------------------------------------------
# Target: aydin-studio — GUI via Xpra (browser access on port 9876)
# ------------------------------------------------------------------------------
FROM base AS aydin-studio

# Install Xpra, Xvfb, and supporting X11 packages for the virtual display
RUN apt-get update && apt-get install -y --no-install-recommends \
        xpra \
        xvfb \
        xterm \
        # Xpra runtime needs: jQuery for HTML5 client, Pillow for window icons
        libjs-jquery \
        libjs-jquery-ui \
        python3-pil \
        # Additional X11 libs for full Qt rendering
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libxtst6 \
        libxi6 \
        x11-xkb-utils \
        xauth \
    && rm -rf /var/lib/apt/lists/*

# Xpra configuration
ENV DISPLAY=:100
ENV XPRA_PORT=9876
ENV XPRA_SCREEN=1920x1080x24+32

# Copy startup script
COPY docker/xpra-start.sh /usr/local/bin/xpra-start.sh
RUN chmod +x /usr/local/bin/xpra-start.sh

EXPOSE 9876

LABEL org.opencontainers.image.title="Aydin Studio" \
      org.opencontainers.image.description="Self-supervised image denoising — GUI (browser access)" \
      org.opencontainers.image.url="https://github.com/royerlab/aydin" \
      org.opencontainers.image.source="https://github.com/royerlab/aydin"

# /data is the default volume mount for user images
VOLUME ["/data"]

CMD ["/usr/local/bin/xpra-start.sh"]
