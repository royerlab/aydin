# Aydin Docker Distribution

Run Aydin without installing Python or any dependencies. Three image variants are available:

| Image | Use case | Size |
|-------|----------|------|
| `aydin` | CLI batch processing (CPU) | ~1.5 GB |
| `aydin-gpu` | CLI with NVIDIA GPU acceleration | ~5 GB |
| `aydin-studio` | Full GUI in your web browser | ~2 GB |

**Important: Memory requirements.** Aydin routinely needs 16-32 GB of RAM for typical microscopy images, and more for very large datasets. Most container runtimes limit memory by default — you **must** increase this or Aydin will be killed mid-processing. See [Troubleshooting: Out of memory](#out-of-memory--container-killed) below.

## When to Use Docker vs Native Install

| | Docker | Native (`pip install aydin`) |
|---|--------|---------------------------|
| **Best for** | Remote servers, HPC, CI/CD, browser-based GUI | Daily work on your own machine |
| **Key advantage** | Run on a powerful server, access GUI from any browser | Best performance, GPU support on all platforms |
| **Speed** | Full native speed on Linux amd64 | Full native speed everywhere |
| **Apple Silicon Mac** | Runs under amd64 emulation (5-10x slower) | Full native speed |
| **Setup** | Just `docker run` — nothing else to install | Requires Python, pip, system libraries |
| **GPU** | NVIDIA on Linux only | NVIDIA (CUDA) or Apple (MPS) |

**Why the emulation penalty on Apple Silicon?** PyQt6 (used by Aydin's GUI and napari) does not publish pre-built wheels for `linux/arm64`. The Docker images must therefore target `linux/amd64`, and on Apple Silicon Macs the container runs under QEMU emulation. This is transparent but significantly slower — fine for quick tests and the GUI, but not practical for denoising large images. On native amd64 Linux (servers, HPC, CI), there is no emulation and Docker runs at full speed.

**Build times** are also affected: the first `docker build` on Apple Silicon takes 15-25 minutes due to emulation. Subsequent builds use Docker's layer cache and take seconds (unless you change `pyproject.toml` or source code). On a native amd64 machine, the first build takes 3-5 minutes.

## Prerequisites: Installing a Container Runtime

You need a container runtime that provides the `docker` CLI. We recommend **OrbStack** on macOS; on Linux, use Docker Engine directly.

### macOS: OrbStack (recommended)

[OrbStack](https://orbstack.dev/) is a fast, lightweight Docker runtime for macOS. Free for open-source and personal use.

```bash
brew install orbstack
```

Open OrbStack from Applications to complete setup. Once the icon appears in the menu bar, `docker` commands work immediately.

**Configure memory:** OrbStack > Settings > Resource Limits > set memory to at least **16 GB**.

**Why OrbStack over Docker Desktop:**

| | OrbStack | Docker Desktop |
|---|----------|---------------|
| Speed | Significantly faster on Apple Silicon | Slower VM overhead |
| Memory | Dynamic allocation, efficient | Fixed allocation, wasteful |
| Disk | Smaller footprint | Larger VM disk image |
| Startup | Near-instant | Several seconds |
| License | Free for personal/open-source | Requires paid subscription for companies >250 employees or >$10M revenue |

### macOS: Colima (open-source alternative)

[Colima](https://github.com/abiosoft/colima) is fully open-source (MIT). Use it if you need a no-restrictions license.

```bash
brew install colima docker
colima start --memory 16 --cpu 4
```

### macOS: Docker Desktop

[Docker Desktop](https://www.docker.com/products/docker-desktop/) works but requires a paid subscription for large organizations.

```bash
brew install --cask docker
```

**Configure memory:** Docker Desktop > Settings > Resources > Memory > set to at least **16 GB**.

### Linux

Install Docker Engine directly (free, open-source):

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
# Log out and back in for group membership to take effect
```

No memory configuration needed — containers use host memory by default.

### Windows

Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) or use WSL2 with Docker Engine. Configure memory in Docker Desktop > Settings > Resources.

---

## Quick Start

### Denoise an image (CLI)

```bash
docker run --rm -v $(pwd):/data ghcr.io/royerlab/aydin \
    denoise /data/noisy_image.tif
```

The denoised result is saved next to the input file.

### Launch Aydin Studio (GUI)

```bash
docker run --rm -p 9876:9876 --shm-size=256m -v $(pwd):/data ghcr.io/royerlab/aydin-studio
```

Then open **http://localhost:9876** in your browser. Aydin Studio appears as an interactive application right in the browser window.

### Denoise with GPU acceleration

```bash
docker run --rm --gpus all -v $(pwd):/data ghcr.io/royerlab/aydin:gpu \
    denoise /data/noisy_image.tif -d noise2selfcnn-unet
```

---

## CLI Usage

The `aydin` CLI provides several commands for image denoising and analysis.

### Basic denoising

```bash
# Denoise with default method (auto-selected)
docker run --rm -v $(pwd):/data ghcr.io/royerlab/aydin \
    denoise /data/image.tif

# Specify output directory
docker run --rm -v $(pwd):/data ghcr.io/royerlab/aydin \
    denoise /data/image.tif --output-folder /data/output
```

### Image quality metrics

```bash
# Compute PSNR between two images
docker run --rm -v $(pwd):/data ghcr.io/royerlab/aydin \
    psnr /data/original.tif /data/denoised.tif

# Compute SSIM
docker run --rm -v $(pwd):/data ghcr.io/royerlab/aydin \
    ssim /data/original.tif /data/denoised.tif
```

### View image metadata

```bash
docker run --rm -v $(pwd):/data ghcr.io/royerlab/aydin \
    info /data/image.tif
```

### Batch processing

Process all TIFF files in a directory:

```bash
for f in data/*.tif; do
    docker run --rm -v $(pwd)/data:/data ghcr.io/royerlab/aydin \
        denoise "/data/$(basename $f)"
done
```

### All available commands

```bash
docker run --rm ghcr.io/royerlab/aydin --help
```

---

## GPU Setup

GPU acceleration requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host machine.

### Install NVIDIA Container Toolkit (Linux)

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verify GPU access

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

### Use a specific GPU

```bash
docker run --rm --gpus '"device=0"' -v $(pwd):/data ghcr.io/royerlab/aydin:gpu \
    denoise /data/image.tif -d noise2selfcnn-unet
```

> **Note:** GPU Docker images only work on Linux hosts with NVIDIA GPUs. macOS and Windows Docker Desktop do not support GPU passthrough.

---

## Aydin Studio (GUI)

Aydin Studio runs inside a virtual display on the server and is streamed to your browser via [Xpra](https://xpra.org/). The primary use case is **running Aydin on a powerful remote machine** (lab workstation, cloud server, HPC node) and accessing the GUI from your laptop.

```
┌─────────────────────┐              ┌──────────────────────────────┐
│  Your laptop         │   browser   │  Powerful server (Linux)     │
│  (any OS)            │────────────▶│  Docker + GPU + lots of RAM  │
│                      │  port 9876  │  aydin-studio container      │
└─────────────────────┘              └──────────────────────────────┘
```

### Local access

Run on the same machine you're browsing from:

```bash
docker run --rm -p 9876:9876 --shm-size=256m -v $(pwd):/data ghcr.io/royerlab/aydin-studio
```

Open **http://localhost:9876** in your browser.

### Remote access (recommended setup)

Run Aydin Studio on a powerful remote Linux server, access it from your laptop:

**On the server:**

```bash
# Start Aydin Studio with your data directory mounted
docker run -d --rm -p 9876:9876 --shm-size=256m \
    -v /path/to/images:/data \
    ghcr.io/royerlab/aydin-studio
```

**On your laptop:**

Open `http://SERVER_IP:9876` in your browser (replace `SERVER_IP` with the server's IP address or hostname).

### Remote access with SSH tunnel (more secure)

If the server is behind a firewall or you want encrypted access, use an SSH tunnel instead of exposing the port directly:

**On your laptop:**

```bash
# Create an SSH tunnel: forwards your local port 9876 to the server's port 9876
ssh -N -L 9876:localhost:9876 user@SERVER_IP
```

Then open **http://localhost:9876** in your browser. Traffic is encrypted through SSH.

### Remote access with GPU

For CNN-based denoising on a remote GPU server, combine with the GPU image or pass `--gpus` to the studio container:

```bash
# On the server (with NVIDIA GPU)
docker run -d --rm -p 9876:9876 --shm-size=256m --gpus all \
    -v /path/to/images:/data \
    ghcr.io/royerlab/aydin-studio
```

### Custom resolution

Match the resolution to your browser window for the best experience:

```bash
docker run --rm -p 9876:9876 --shm-size=256m \
    -e XPRA_SCREEN=2560x1440x24+32 \
    -v /path/to/images:/data \
    ghcr.io/royerlab/aydin-studio
```

### Custom port

```bash
docker run --rm -p 8080:8080 --shm-size=256m \
    -e XPRA_PORT=8080 \
    -v /path/to/images:/data \
    ghcr.io/royerlab/aydin-studio
```

---

## Getting Your Data In and Out

This applies to **all** Docker variants (CLI, GPU, and Studio).

Docker containers are isolated from the host filesystem. The `-v` flag mounts a host directory into the container at `/data`. **Only files inside `/data` are visible to the container, and only files written to `/data` are saved back to your machine.**

```
Host machine                     Container
─────────────                    ─────────
/home/user/microscopy/  ──────▶  /data/
  ├── sample.tif                   ├── sample.tif        (read from here)
  └── sample_denoised.tif  ◀────  └── sample_denoised.tif (written here)
```

**CLI example:**

```bash
# Mount your images directory, denoise, output appears on host
docker run --rm -v /home/user/microscopy:/data ghcr.io/royerlab/aydin \
    denoise /data/sample.tif
# Result: /home/user/microscopy/sample_denoised.tif
```

**Studio GUI example:**

```bash
# Start Studio with your images mounted
docker run --rm -p 9876:9876 --shm-size=256m \
    -v /home/user/microscopy:/data \
    ghcr.io/royerlab/aydin-studio
```

In Aydin Studio, use **File > Open** and browse to `/data/` to load your images. When denoising completes, the output is saved alongside the input in `/data/` — which means it appears on your host machine automatically.

**Important:** Aydin's built-in sample datasets (File > Open Sample) download to `/root/.cache/` inside the container, which is **not** mounted. Results from sample datasets are lost when the container stops. To keep them, either:
- Copy them out: `docker cp CONTAINER_ID:/root/.cache/data/ ./samples/`
- Or download the samples to your mounted `/data/` directory first

**macOS/Windows shorthand** — mount the current directory:

```bash
docker run --rm -v $(pwd):/data ghcr.io/royerlab/aydin denoise /data/image.tif
```

**Multiple directories** (read-only input, writable output):

```bash
docker run --rm \
    -v /path/to/input:/input:ro \
    -v /path/to/output:/output \
    ghcr.io/royerlab/aydin \
    denoise /input/image.tif --output-folder /output
```

---

## Docker Compose

A `docker-compose.yml` is provided in the repository root. By default it mounts `./data/` from the repo directory.

```bash
# Place images in ./data/, then:

# CLI denoising
docker compose run --rm aydin denoise /data/image.tif

# CLI with GPU
docker compose run --rm aydin-gpu denoise /data/image.tif

# Launch Studio GUI
docker compose up aydin-studio
# Open http://localhost:9876, load images from /data/
```

---

## Building from Source

Clone the repository and build locally:

```bash
git clone https://github.com/royerlab/aydin.git
cd aydin

# Build all variants (recommended — handles platform automatically)
make docker-build

# Or build individually (--platform required on Apple Silicon)
docker build --platform linux/amd64 --target aydin        -t aydin .
docker build --platform linux/amd64 --target aydin-gpu    -t aydin-gpu .
docker build --platform linux/amd64 --target aydin-studio -t aydin-studio .
```

On native Linux amd64, you can omit `--platform linux/amd64`. The Makefile handles this automatically via the `DOCKER_PLATFORM` variable.

---

## HPC / Singularity / Apptainer

Docker images can be converted to Singularity/Apptainer format for use on HPC clusters:

```bash
# Convert to Singularity Image Format (SIF)
singularity pull docker://ghcr.io/royerlab/aydin:latest

# Run on HPC
singularity run aydin_latest.sif denoise image.tif

# With GPU (NVIDIA)
singularity run --nv aydin_latest.sif denoise image.tif -d noise2selfcnn-unet
```

---

## Troubleshooting

### "Permission denied" on output files

Docker may create output files as root. Fix with `--user`:

```bash
docker run --rm --user $(id -u):$(id -g) -v $(pwd):/data ghcr.io/royerlab/aydin \
    denoise /data/image.tif
```

### GUI: crash or SIGBUS error

Qt applications need more shared memory than Docker's default 64 MB. All GUI examples above include `--shm-size=256m`. If you see crashes or `SIGBUS` errors, increase it:

```bash
docker run --rm -p 9876:9876 --shm-size=512m -v $(pwd):/data ghcr.io/royerlab/aydin-studio
```

### GUI: blank screen or "connection refused"

- Ensure port 9876 is not in use: `lsof -i :9876`
- Wait a few seconds after starting — Xpra needs time to initialize
- Try a different browser if the HTML5 client doesn't load

### GPU: "could not select device driver"

- Verify NVIDIA Container Toolkit is installed: `nvidia-ctk --version`
- Restart Docker after installing: `sudo systemctl restart docker`
- Check GPU visibility: `docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi`

### Out of memory / container killed

Aydin typically needs **16-32 GB of RAM** for microscopy images. Most container runtimes limit memory by default, which will cause Aydin to be silently killed (exit code 137) during training or inference.

**OrbStack (macOS):**
OrbStack > Settings > Resource Limits > set memory to at least **16 GB** (32 GB recommended).

**Docker Desktop (macOS / Windows):**
Docker Desktop > Settings > Resources > Memory > set to at least **16 GB** (32 GB recommended) > Apply & restart.

**Colima (macOS):**
```bash
colima stop && colima start --memory 32 --cpu 4
```

**Linux (native Docker):**
Containers use host memory by default — no limit unless explicitly set. If you do set a limit, ensure it is generous:

```bash
docker run --rm --memory=32g -v $(pwd):/data ghcr.io/royerlab/aydin denoise /data/image.tif
```

**Symptoms of insufficient memory:**
- Container exits silently with code 137 (OOM killed)
- Process hangs during feature generation or training
- `dmesg | grep -i oom` on Linux shows OOM killer messages
