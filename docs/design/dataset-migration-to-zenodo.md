# Migrating Aydin Example Datasets from Google Drive to Zenodo

## Motivation

The Aydin example datasets are currently hosted on Google Drive and downloaded via
`gdown`. This is brittle:

- `gdown` frequently breaks when Google changes their download page or captcha flow
- Google Drive file IDs can stop working without warning
- No checksums, no DOI, no citation metadata
- Adds `gdown` as an unnecessary dependency

**Zenodo** (backed by CERN) provides permanent DOIs, stable REST API downloads,
per-file access with checksums, and no authentication for public records.

---

## Dataset Inventory

### 35 datasets total, ~7.5 GB uncompressed

| Category | Count | Raw Size | Best Compressed | Notes |
|----------|-------|----------|-----------------|-------|
| Used in tests | 15 | ~3.3 GB | ~1.6 GB | Must keep |
| Demo-only | 7 | ~3.3 GB | ~1.6 GB | Keep if demos matter |
| GUI-menu-only | 6 | ~84 MB | ~45 MB | Keep for GUI UX |
| Currently unreferenced | 6 | ~669 MB | ~340 MB | Keep — potentially useful |

### Currently unreferenced datasets (kept for future use)

These 6 enum members are not currently referenced in code but are kept as
potentially useful example data:

| Dataset | Size | License | Hosting Decision |
|---------|------|---------|------------------|
| `leonetti_tm7sf2` | 141 MB | CC BY-SA 4.0 (OpenCell) | **Zenodo** |
| `leonetti_sptssa` | 145 MB | CC BY-SA 4.0 (OpenCell) | **Zenodo** |
| `cognet_nanotube1` | 249 MB | PI-confirmed safe | **Zenodo** |
| `ome_mitocheck` | 127 MB | Public Domain (OME Consortium) | **Source URL** (openmicroscopy.org) |
| `ome_spim` | 6.9 MB | CC BY 4.0 (OME Consortium) | **Source URL** (openmicroscopy.org) |
| `generic_crowd` | 257 KB | **Unknown** (USC-SIPI, removed from database) | **Dropped** — no stable source URL, never referenced |

---

## Compression Results

All large TIFFs are stored **uncompressed** (TIFF compression=1). Experiments with
multiple algorithms show that **TIFF with zstd compression + horizontal predictor**
provides the best tradeoff of ratio, speed, and format compatibility (readable by
tifffile, ImageJ/FIJI, and most modern TIFF libraries).

| Dataset | Raw | Compressed (TIFF zstd+pred) | Ratio | Notes |
|---------|-----|---------------------------|-------|-------|
| Cognet 400fps | 1563 MB | ~797 MB | 2.0x | uint16, narrow dynamic range |
| Flybrain | 992 MB | ~501 MB | 2.0x | uint16, 12-bit effective |
| Hyman HeLa | 358 MB | ~120 MB | 3.0x | uint16, very sparse (max=927) |
| Leonetti SNCA | 184 MB | ~139 MB | 1.3x | uint16, high dynamic range |
| Keller Dmel | 149 MB | ~est. 100 MB | 1.5x | uint16 |
| Leonetti ARHGAP21 | 73 MB | ~est. 55 MB | 1.3x | uint16 |
| Leonetti ANKRD11 | 73 MB | ~est. 55 MB | 1.3x | uint16 |
| Maitre mouse | 76 MB | ~55 MB | 1.4x | uint8 |
| Royer HCR | 63 MB | ~45 MB | 1.4x | uint16, 14-bit |
| Tribolium | 47 MB | ~18 MB | 2.6x | uint16 but max=254 |

**Estimated total after compression: ~3.0-3.5 GB** (within Zenodo's 50 GB limit).

The small files (PNG, JPG, < 5 MB) are already compressed and uploaded as-is.

> **Note:** For maximum compression, `blosc(cname='zstd', shuffle=SHUFFLE)` via zarr
> gives ~5-10% better ratios, but changes the format. TIFF zstd is preferred for
> universal compatibility and individual-file access on Zenodo.

---

## Licensing Analysis

### Clearly licensed (safe to redistribute)

| Dataset | Source | License | Compatible with CC-BY-4.0? |
|---------|--------|---------|---------------------------|
| Myers Tribolium | CARE paper (MPI-CBG) | **CC0 1.0** (public domain) | Yes |
| Machado egg chamber | LimeSeg (Zenodo) | **CC BY 4.0** | Yes |
| Brown SIDD chessboard | SIDD dataset (Brown U.) | **MIT License** | Yes |
| Mona Lisa (noisy) | Public domain painting | **Public Domain** | Yes |
| OpenCell / Leonetti (3 datasets) | CZ Biohub | **CC BY-SA 4.0** | Yes (ShareAlike) |

### Collaborator-provided (permission confirmed by PI)

These datasets were shared directly with the Aydin project. The PI confirms
redistribution is permitted:

| Dataset | Source Lab |
|---------|-----------|
| Keller Dmel | Keller Lab, Janelia Research Campus |
| Janelia Flybrain | Janelia Research Campus |
| Cognet nanotubes (3 files) | Cognet Lab, Univ. Bordeaux |
| Maitre mouse blastocyst | Maitre Lab, Institut Curie |
| Hyman HeLa | Hyman Lab, MPI-CBG |
| Huang fixed pattern noise | Huang Lab |
| Royer HCR | Royer Lab (own data) |

### Small images — provenance resolved

| Dataset | Source | License | CC-BY-4.0 OK? |
|---------|--------|---------|---------------|
| `newyork.png` | Royer (own photograph) | Owner-released | **YES** |
| `scafoldings.png` | Royer (own photograph) | Owner-released | **YES** |
| `andromeda.png` | Isaac Roberts, 1888 photograph of M31 | **Public Domain** (copyright expired, photographer died 1904) | **YES** |
| `characters.jpg` | Ancient writing comparison chart (Cuneiform/Egyptian/Chinese) | **Public Domain** (Wikimedia Commons) | **YES** |
| `Gauss_noisy.png` | Portrait of C.F. Gauss by C.A. Jensen, 1840 (with added noise) | **Public Domain** (artist died 1870) | **YES** |
| `monalisa.png` | Leonardo da Vinci painting (with added noise) | **Public Domain** | **YES** |
| `pollen.png` | SEM micrograph, Dartmouth College (Louisa Howard & Charles Daghlian) | **Public Domain** (with attribution to Dartmouth) | **YES** |
| `periodic_noise.png` | OpenCV documentation sample (newspaper/magazine scan with periodic noise) | **Apache 2.0 / BSD-3** (OpenCV) | **YES** |
| `fountain.png` | Wallace fountain / "sta1" from U. Edinburgh image processing library | **Open Access / Public Domain** (19th-c. sculpture, with attribution) | **YES** |
| `Brown_SIDD_chessboard_gray.png` | SIDD dataset (Abdelhamed et al., CVPR 2018, Brown U.) | **MIT License** | **YES** |
| `rgbtest.png` | Test pattern (project-created) | Project-owned | **YES** |
| `newyork_noisy.tif` | Derived from newyork.png (Royer, own photograph + synthetic noise) | Owner-released | **YES** |

### Problematic — not hosted on Zenodo (source-URL download instead)

| Dataset | Source | License | Issue | Action |
|---------|--------|---------|-------|--------|
| `camera.png` | MIT "Cameraman" (Schreiber, 1978) | **CC BY-NC** | Non-commercial restriction incompatible with CC-BY-4.0 | **Source URL**: `dome.mit.edu` |
| `mandrill.tif` | USC-SIPI database (image 4.2.03) | **Copyright unknown** | Cannot redistribute | **Source URL**: `sipi.usc.edu` (confirmed working) |
| `lizard.png` | Berkeley BSDS300 dataset, test image #87046 (zebra-tailed lizard) | **Non-commercial research only** | Incompatible with CC-BY-4.0 | **Source URL**: `eecs.berkeley.edu` |

### Recommended Zenodo license

Use **CC-BY-4.0** as the overarching license, with per-file attribution notes in
the description for datasets with specific licenses (CC BY-SA 4.0 for OpenCell,
MIT for SIDD, CC0 for CARE/Tribolium, CC BY-NC-SA for Maitre blastocyst).

Note: The Maitre blastocyst data is CC BY-NC-SA per its original publication
(Dumortier et al., Science 2019). This is technically incompatible with
CC-BY-4.0, but the PI has confirmed redistribution is permitted. Document this
in the Zenodo record notes.

---

## Hybrid Download Strategy

Not all datasets can (or should) be hosted on Zenodo. The solution is a
**hybrid approach** with three download backends:

### Backend 1: Zenodo (primary — most datasets)

All datasets with compatible licenses are uploaded to a single Zenodo record
and downloaded via:
```
GET https://zenodo.org/api/records/{RECORD_ID}/files/{FILENAME}/content
```

### Backend 2: Original source URLs (license-incompatible or externally-hosted images)

Some images cannot or should not be hosted on Zenodo — either because their
license prevents redistribution under CC-BY-4.0, or because a stable
authoritative source URL already exists. We download these directly from
their canonical source URLs at runtime:

| Image | Source URL | License | Notes |
|-------|-----------|---------|-------|
| `camera.png` | `https://dome.mit.edu/bitstream/handle/1721.3/195767/cameraman.tif` | CC BY-NC (MIT) | Original 256x256 TIFF; resize to 512x512 on download to match current behavior |
| `lizard.png` | `https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/plain/normal/gray/87046.jpg` | Non-commercial research (Berkeley BSDS300) | 481x321 JPEG, grayscale. Used in 3 tests + 1 demo |
| `mandrill.tif` | `https://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03` | Copyright unknown (USC-SIPI) | 512x512 RGB TIFF. Only used in 1 demo (`demo_datasets.py`) |
| `ome_mitocheck` | `https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/MitoCheck/00001_01.ome.tiff` | Public Domain (OME Consortium) | OME-TIFF, currently unreferenced but kept for future use |
| `ome_spim` | `https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/modulo/SPIM-ModuloAlongZ.ome.tiff` | CC BY 4.0 (OME Consortium) | OME-TIFF, currently unreferenced but kept for future use |

The first three URLs are hosted by MIT, UC Berkeley, and USC — institutions
that have maintained these URLs for 20+ years. The OME Consortium URLs are
maintained by the Open Microscopy Environment project (University of Dundee).

### Dropped: `crowd.tif`

`generic_crowd` (512x512 uint8, 257 KB) was sourced from the USC-SIPI Miscellaneous
image database (`misc/5.1.01`). **USC-SIPI has removed this file** from their
database — the download URL returns "File not found". No alternative stable source
URL was found (checked University of Waterloo, University of Cape Town, University
of Granada repositories). The image's license and original copyright holder are
unknown.

**Decision: Drop.** The image is never referenced in the codebase, has unknown
licensing, and no stable source URL exists. The enum member `generic_crowd` is
removed.

### Implementation in `datasets.py`

The `examples_single` enum values change from `(gdrive_id, filename)` tuples
to a new format that encodes the backend:

```python
ZENODO_RECORD_ID = "XXXXXXX"  # filled in after publishing

class examples_single(Enum):
    def get_path(self):
        source, filename = self.value
        if source == "zenodo":
            download_from_zenodo(filename, datasets_folder)
        elif source.startswith("http"):
            download_from_url(source, filename, datasets_folder)
        return join(datasets_folder, filename)

    def get_array(self):
        array, _ = io.imread(self.get_path())
        return array

    # --- Zenodo-hosted (CC-BY-4.0 compatible) ---
    generic_newyork = ("zenodo", "newyork.png")
    generic_pollen = ("zenodo", "pollen.png")
    generic_scafoldings = ("zenodo", "scafoldings.png")
    generic_andromeda = ("zenodo", "andromeda.png")
    generic_characters = ("zenodo", "characters.jpg")
    noisy_fountain = ("zenodo", "fountain.png")
    noisy_newyork = ("zenodo", "newyork_noisy.tif")
    noisy_monalisa = ("zenodo", "monalisa.png")
    noisy_gauss = ("zenodo", "Gauss_noisy.png")
    noisy_brown_chessboard = ("zenodo", "Brown_SIDD_chessboard_gray.png")
    periodic_noise = ("zenodo", "periodic_noise.png")
    rgbtest = ("zenodo", "rgbtest.png")
    leonetti_snca = ("zenodo", "Leonetti_p1H8_2_SNCA_PyProcessed_IJClean.tif")
    leonetti_arhgap21 = ("zenodo", "Leonetti_OC-FOV_ARHGAP21_ENSG00000107863_CID000556_FID00030711_stack.tif")
    leonetti_ankrd11 = ("zenodo", "Leonetti_OC-FOV_ANKRD11_ENSG00000167522_CID001385_FID00033338_stack.tif")
    huang_fixed_pattern_noise = ("zenodo", "fixed_pattern_noise.tif")
    keller_dmel = ("zenodo", "SPC0_TM0132_CM0_CM1_CHN00_CHN01.fusedStack.tif")
    janelia_flybrain = ("zenodo", "Flybrain_3ch_mediumSize.tif")
    myers_tribolium = ("zenodo", "Myers_Tribolium_nGFP_0.1_0.2_0.5_20_13_late.tif")
    royerlab_hcr = ("zenodo", "Royer_confocal_dragonfly_hcr_drerio_30somite_crop.tif")
    machado_drosophile_egg_chamber = ("zenodo", "C2-DrosophilaEggChamber-small.tif")
    cognet_nanotube_400fps = ("zenodo", "Cognet_1-400fps.tif")
    cognet_nanotube_200fps = ("zenodo", "Cognet_1-200fps.tif")
    cognet_nanotube_100fps = ("zenodo", "Cognet_1-100fps.tif")
    maitre_mouse = ("zenodo", "Maitre_mouse blastocyst_fracking_180124_e3_crop.tif")
    hyman_hela = ("zenodo", "Hyman_HeLa.tif")
    leonetti_tm7sf2 = ("zenodo", "Leonetti_p4B3_1_TM7SF2_PyProcessed_IJClean.tif")
    leonetti_sptssa = ("zenodo", "Leonetti_p1H10_2_SPTSSA_PyProcessed_IJClean.tif")
    cognet_nanotube1 = ("zenodo", "Cognet_r03-s01-100mW-20ms-175 50xplpeg-173.tif")

    # --- Source-URL-hosted (license-incompatible or externally-hosted) ---
    generic_camera = (
        "https://dome.mit.edu/bitstream/handle/1721.3/195767/cameraman.tif",
        "camera.tif",
    )
    generic_lizard = (
        "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/"
        "BSDS300/html/images/plain/normal/gray/87046.jpg",
        "lizard.jpg",
    )
    generic_mandrill = (
        "https://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03",
        "mandrill.tif",
    )
    ome_mitocheck = (
        "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/"
        "MitoCheck/00001_01.ome.tiff",
        "00001_01.ome.tiff",
    )
    ome_spim = (
        "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/"
        "modulo/SPIM-ModuloAlongZ.ome.tiff",
        "SPIM-ModuloAlongZ.ome.tiff",
    )

    # DROPPED: crowd.tif — unknown license, removed from USC-SIPI database,
    # no stable source URL found, never referenced in codebase.
```

The two download functions:

```python
def download_from_zenodo(filename, dest_folder=datasets_folder, overwrite=False):
    """Download a single file from the Aydin Zenodo dataset record."""
    output_path = join(dest_folder, filename)
    if not overwrite and exists(output_path):
        aprint(f"Not downloading {filename} as it already exists.")
        return None
    os.makedirs(dest_folder, exist_ok=True)
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/{filename}/content"
    aprint(f"Downloading {filename} from Zenodo...")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    return output_path


def download_from_url(url, filename, dest_folder=datasets_folder, overwrite=False):
    """Download a file from a direct URL."""
    output_path = join(dest_folder, filename)
    if not overwrite and exists(output_path):
        aprint(f"Not downloading {filename} as it already exists.")
        return None
    os.makedirs(dest_folder, exist_ok=True)
    aprint(f"Downloading {filename} from {url}...")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    return output_path
```

---

## Step-by-Step Migration Guide

### Prerequisites

1. A [Zenodo account](https://zenodo.org/signup/)
2. A personal access token with `deposit:write` and `deposit:actions` scopes
   (create at https://zenodo.org/account/settings/applications/)
3. All dataset files cached locally (run `aydin.io.datasets.download_all_examples()`)
4. Python with `requests` and `tifffile` installed

### Step 1: Prepare

- [ ] Remove `generic_crowd` from `examples_single` enum (no stable source, unknown license)
- [ ] Verify the 5 source URLs still work (camera, lizard, mandrill, ome_mitocheck, ome_spim)
- [ ] Add `leonetti_tm7sf2`, `leonetti_sptssa`, `cognet_nanotube1` to the Zenodo file list
- [ ] Add `ome_mitocheck` and `ome_spim` as source-URL entries

### Step 2: Compress large TIFFs

```python
import tifffile
import os

CACHE_DIR = os.path.expanduser("~/.cache/data")
OUTPUT_DIR = "/tmp/aydin_zenodo_upload"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Files > 10 MB get re-compressed; small files copied as-is
LARGE_TIFFS = [
    "Hyman_HeLa.tif",
    "Flybrain_3ch_mediumSize.tif",
    "Cognet_1-400fps.tif",
    "Cognet_1-200fps.tif",
    "Cognet_1-100fps.tif",
    "Cognet_r03-s01-100mW-20ms-175 50xplpeg-173.tif",
    "Leonetti_p1H8_2_SNCA_PyProcessed_IJClean.tif",
    "Leonetti_OC-FOV_ARHGAP21_ENSG00000107863_CID000556_FID00030711_stack.tif",
    "Leonetti_OC-FOV_ANKRD11_ENSG00000167522_CID001385_FID00033338_stack.tif",
    "Leonetti_p4B3_1_TM7SF2_PyProcessed_IJClean.tif",
    "Leonetti_p1H10_2_SPTSSA_PyProcessed_IJClean.tif",
    "SPC0_TM0132_CM0_CM1_CHN00_CHN01.fusedStack.tif",
    "Myers_Tribolium_nGFP_0.1_0.2_0.5_20_13_late.tif",
    "Royer_confocal_dragonfly_hcr_drerio_30somite_crop.tif",
    "C2-DrosophilaEggChamber-small.tif",
    "Maitre_mouse blastocyst_fracking_180124_e3_crop.tif",
    "fixed_pattern_noise.tif",
    "newyork_noisy.tif",
]

for fname in LARGE_TIFFS:
    src = os.path.join(CACHE_DIR, fname)
    dst = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(src):
        print(f"  SKIP (not found): {fname}")
        continue
    print(f"Compressing {fname}...")
    data = tifffile.imread(src)
    tifffile.imwrite(dst, data, compression='zstd', predictor=True)
    src_mb = os.path.getsize(src) / (1024 * 1024)
    dst_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"  {src_mb:.1f} MB -> {dst_mb:.1f} MB ({src_mb/dst_mb:.1f}x)")
    del data
```

### Step 3: Create Zenodo deposition and upload files

```python
import os
import shutil
import requests

ZENODO_TOKEN = os.environ["ZENODO_TOKEN"]
ZENODO_API = "https://zenodo.org/api"
PARAMS = {"access_token": ZENODO_TOKEN}
HEADERS = {"Content-Type": "application/json"}

CACHE_DIR = os.path.expanduser("~/.cache/data")
COMPRESSED_DIR = "/tmp/aydin_zenodo_upload"

# Files to upload to Zenodo (only license-compatible ones)
# camera.png, lizard.png, mandrill.tif are NOT uploaded — they are
# downloaded from their original source URLs at runtime instead.
# ome_mitocheck and ome_spim are also NOT uploaded — they are
# downloaded from openmicroscopy.org at runtime.
ALL_FILES = [
    # Small files (upload as-is from cache)
    "newyork.png",
    "pollen.png",
    "scafoldings.png",
    "andromeda.png",
    "characters.jpg",
    "rgbtest.png",
    "fountain.png",
    "monalisa.png",
    "Gauss_noisy.png",
    "Brown_SIDD_chessboard_gray.png",
    "periodic_noise.png",
    # Large files (upload from compressed dir)
    "Hyman_HeLa.tif",
    "Flybrain_3ch_mediumSize.tif",
    "Cognet_1-400fps.tif",
    "Cognet_1-200fps.tif",
    "Cognet_1-100fps.tif",
    "Cognet_r03-s01-100mW-20ms-175 50xplpeg-173.tif",
    "Leonetti_p1H8_2_SNCA_PyProcessed_IJClean.tif",
    "Leonetti_OC-FOV_ARHGAP21_ENSG00000107863_CID000556_FID00030711_stack.tif",
    "Leonetti_OC-FOV_ANKRD11_ENSG00000167522_CID001385_FID00033338_stack.tif",
    "Leonetti_p4B3_1_TM7SF2_PyProcessed_IJClean.tif",
    "Leonetti_p1H10_2_SPTSSA_PyProcessed_IJClean.tif",
    "SPC0_TM0132_CM0_CM1_CHN00_CHN01.fusedStack.tif",
    "Myers_Tribolium_nGFP_0.1_0.2_0.5_20_13_late.tif",
    "Royer_confocal_dragonfly_hcr_drerio_30somite_crop.tif",
    "C2-DrosophilaEggChamber-small.tif",
    "Maitre_mouse blastocyst_fracking_180124_e3_crop.tif",
    "fixed_pattern_noise.tif",
    "newyork_noisy.tif",
]

# --- Create deposition ---
r = requests.post(f"{ZENODO_API}/deposit/depositions",
                   params=PARAMS, json={}, headers=HEADERS)
r.raise_for_status()
deposition = r.json()
dep_id = deposition["id"]
bucket_url = deposition["links"]["bucket"]
print(f"Created deposition: {dep_id}")
print(f"  Edit at: https://zenodo.org/deposit/{dep_id}")

# --- Upload each file individually ---
for fname in ALL_FILES:
    # Prefer compressed version if it exists
    compressed = os.path.join(COMPRESSED_DIR, fname)
    original = os.path.join(CACHE_DIR, fname)
    upload_path = compressed if os.path.exists(compressed) else original

    if not os.path.exists(upload_path):
        print(f"  SKIP: {fname} (not found)")
        continue

    size_mb = os.path.getsize(upload_path) / (1024 * 1024)
    print(f"  Uploading {fname} ({size_mb:.1f} MB)...")

    with open(upload_path, "rb") as fp:
        r = requests.put(f"{bucket_url}/{fname}", params=PARAMS, data=fp)
        r.raise_for_status()

    checksum = r.json().get("checksum", "?")
    print(f"    OK: {checksum}")

print(f"\nAll files uploaded to deposition {dep_id}")
```

### Step 4: Set metadata

```python
metadata = {
    "metadata": {
        "title": (
            "Aydin Example Datasets: Microscopy and Natural Images "
            "for Image Denoising Benchmarks"
        ),
        "upload_type": "dataset",
        "description": """<p>A curated collection of example images used by
<a href="https://github.com/royerlab/aydin">Aydin</a>, a self-supervised image
denoising toolkit for n-dimensional microscopy and natural images.</p>

<p>This dataset includes 2D natural images, noisy test images, 3D fluorescence
microscopy volumes, 2D+t and 3D+t time-lapse sequences from light-sheet and
confocal microscopy. The images serve as benchmarks for testing and demonstrating
Aydin's classical, feature-based, and CNN denoising algorithms.</p>

<p><strong>Individual file access:</strong> Each file can be downloaded individually
via the Zenodo REST API:<br>
<code>GET https://zenodo.org/api/records/{record_id}/files/{filename}/content</code>
</p>

<p><strong>Large TIFF files</strong> have been losslessly compressed using TIFF zstd
compression with horizontal predictor. They can be read directly by tifffile,
ImageJ/FIJI, and most modern TIFF readers.</p>

<p><strong>Data sources and attribution:</strong></p>
<ul>
<li><strong>New York, Scaffoldings</strong> — original photographs
    (Loic A. Royer), released CC BY 4.0</li>
<li><strong>Andromeda</strong> — Isaac Roberts, 1888 photograph of M31
    (public domain, copyright expired)</li>
<li><strong>Characters</strong> — comparative ancient writing chart
    (public domain, Wikimedia Commons)</li>
<li><strong>Gauss noisy</strong> — portrait of C.F. Gauss by C.A. Jensen, 1840,
    with synthetic noise (public domain)</li>
<li><strong>Mona Lisa noisy</strong> — Leonardo da Vinci painting with synthetic
    noise (public domain)</li>
<li><strong>Pollen</strong> — SEM micrograph by Louisa Howard &amp; Charles Daghlian,
    Dartmouth College (public domain)</li>
<li><strong>Periodic noise</strong> — OpenCV documentation sample
    (Apache 2.0 / BSD-3)</li>
<li><strong>Fountain</strong> — Wallace fountain, U. Edinburgh image processing
    library (public domain / open access)</li>
<li><strong>SIDD chessboard</strong> — from the
    <a href="https://abdokamel.github.io/sidd/">SIDD dataset</a>
    (Abdelhamed et al., CVPR 2018), MIT License</li>
<li><strong>OpenCell / Leonetti</strong> — from the
    <a href="https://opencell.czbiohub.org/">OpenCell</a> project
    (Chan Zuckerberg Biohub), licensed CC BY-SA 4.0</li>
<li><strong>Myers / Tribolium</strong> — from the
    <a href="https://doi.org/10.17617/3.FDFZOF">CARE dataset</a>
    (Weigert et al., Nature Methods 2018), licensed CC0 1.0</li>
<li><strong>Machado / Drosophila egg chamber</strong> — from the
    <a href="https://doi.org/10.5281/zenodo.1472859">LimeSeg dataset</a>
    (Machado et al., BMC Bioinformatics 2019), licensed CC BY 4.0</li>
<li><strong>Brown / SIDD chessboard</strong> — from the
    <a href="https://abdokamel.github.io/sidd/">SIDD dataset</a>
    (Abdelhamed et al., CVPR 2018), licensed MIT</li>
<li><strong>Keller / Drosophila</strong> — light-sheet data
    (Keller Lab, Janelia Research Campus / HHMI)</li>
<li><strong>Janelia Flybrain</strong> — 3-channel fly brain volume
    (Janelia Research Campus / HHMI)</li>
<li><strong>Royer HCR</strong> — confocal HCR zebrafish 30-somite
    (Royer Lab, Chan Zuckerberg Biohub)</li>
<li><strong>Cognet nanotubes</strong> — SWCNT tracking at 100/200/400 fps
    (Cognet Lab, University of Bordeaux)</li>
<li><strong>Maitre blastocyst</strong> — mouse blastocyst time-lapse
    (Maitre Lab, Institut Curie).
    See <a href="https://doi.org/10.1126/science.aaw7709">Dumortier et al.,
    Science 2019</a></li>
<li><strong>Hyman HeLa</strong> — HeLa cell 4D dataset
    (Hyman Lab, MPI-CBG)</li>
<li><strong>Huang</strong> — fixed pattern noise volume</li>
</ul>

<p><strong>Note:</strong> A few additional datasets are not hosted in this Zenodo
record but are downloaded at runtime from their original source URLs:
camera (MIT, CC BY-NC), lizard (Berkeley BSDS300, non-commercial research),
mandrill (USC-SIPI), and OME-TIFF examples from
<a href="https://downloads.openmicroscopy.org/">openmicroscopy.org</a>.</p>""",
        "creators": [
            {
                "name": "Royer, Loic A.",
                "affiliation": "Chan Zuckerberg Biohub",
                "orcid": "0000-0002-9991-4560",
            },
            {
                "name": "Solak, Ahmet Can",
                "affiliation": "Chan Zuckerberg Biohub",
            },
        ],
        "keywords": [
            "image denoising",
            "microscopy",
            "fluorescence microscopy",
            "light-sheet microscopy",
            "confocal microscopy",
            "benchmark datasets",
            "self-supervised learning",
            "aydin",
        ],
        "license": "CC-BY-4.0",
        "access_right": "open",
        "related_identifiers": [
            {
                "identifier": "10.5281/zenodo.5654826",
                "relation": "isSupplementTo",
                "scheme": "doi",
            },
            {
                "identifier": "https://github.com/royerlab/aydin",
                "relation": "isSupplementTo",
                "scheme": "url",
            },
        ],
        "version": "1.0.0",
        "notes": (
            "These datasets are automatically downloaded by the aydin Python "
            "package (aydin.io.datasets module) for testing and demonstration. "
            "Individual files with specific licenses: OpenCell/Leonetti data is "
            "CC BY-SA 4.0; CARE/Tribolium data is CC0 1.0; LimeSeg/Machado data "
            "is CC BY 4.0; SIDD/Brown data is MIT License. Maitre blastocyst "
            "data is CC BY-NC-SA per the original publication."
        ),
    }
}

r = requests.put(
    f"{ZENODO_API}/deposit/depositions/{dep_id}",
    params=PARAMS,
    json=metadata,
    headers=HEADERS,
)
r.raise_for_status()
print("Metadata updated successfully")
```

> **Important:** Update the ORCID for Loic Royer above. The one shown is a
> placeholder — replace with the correct ORCID.

### Step 5: Review before publishing

1. Visit `https://zenodo.org/deposit/{dep_id}` in your browser
2. Verify all files are listed with correct sizes
3. Check the metadata renders correctly (title, description, attribution)
4. Verify the license is CC-BY-4.0
5. Check the related identifiers link to the Aydin software DOI

**Do NOT publish until you are satisfied.** Once published, you cannot delete
files (only add new versions).

### Step 6: Publish

```python
r = requests.post(
    f"{ZENODO_API}/deposit/depositions/{dep_id}/actions/publish",
    params=PARAMS,
)
r.raise_for_status()
record = r.json()
doi = record["doi"]
record_id = record["id"]
print(f"Published! DOI: {doi}")
print(f"Record ID: {record_id}")
print(f"URL: https://zenodo.org/records/{record_id}")
```

Save the `record_id` — this is what goes into `datasets.py`.

### Step 7: Update `aydin/io/datasets.py`

Replace the Google Drive download mechanism with the hybrid Zenodo + source URL
approach described in the "Hybrid Download Strategy" section above.

Key changes:
1. Replace `import gdown` with `import requests`
2. Add `ZENODO_RECORD_ID` constant (from Step 6 output)
3. Add `download_from_zenodo()` and `download_from_url()` functions
4. Replace `download_from_gdrive()` calls — no longer needed
5. Update `examples_single` enum values from `(gdrive_id, filename)` to
   `("zenodo", filename)` or `(url, filename)` format
6. Remove `generic_crowd` enum member (no stable source, unknown license)
7. Add `ome_mitocheck` and `ome_spim` as source-URL entries
   (openmicroscopy.org)
8. Add `leonetti_tm7sf2`, `leonetti_sptssa`, `cognet_nanotube1` as
   Zenodo entries
9. Update convenience functions if filenames changed (e.g., `camera.tif`
   instead of `camera.png`)

See the "Implementation in `datasets.py`" section above for the complete
new enum and download functions.

### Step 8: Remove `gdown` dependency

In `pyproject.toml`, remove `gdown` from `dependencies`.

### Step 9: Test

```bash
# Clear cache to force re-download
rm -rf ~/.cache/data/

# Test Zenodo downloads
hatch run python -c "from aydin.io.datasets import newyork; print(newyork().shape)"

# Test source-URL downloads
hatch run python -c "from aydin.io.datasets import camera; print(camera().shape)"
hatch run python -c "from aydin.io.datasets import lizard; print(lizard().shape)"

# Run the full dataset test suite
hatch run pytest src/aydin/io/tests/test_datasets.py -v --disable-pytest-warnings
```

---

## Zenodo API Quick Reference

### Download a single file (no auth required)
```
GET https://zenodo.org/api/records/{RECORD_ID}/files/{FILENAME}/content
```

### List all files in a record
```
GET https://zenodo.org/api/records/{RECORD_ID}
```
Response includes `files[]` array with `key`, `size`, `checksum`, and download URL.

### File checksums
Each file has an MD5 checksum in the metadata (`files[].checksum`), enabling
integrity verification after download.

---

## Versioning

If datasets are added or updated later, create a **new version** of the record:

```python
r = requests.post(
    f"{ZENODO_API}/deposit/depositions/{dep_id}/actions/newversion",
    params=PARAMS,
)
new_draft = r.json()["links"]["latest_draft"]
# Then modify files/metadata on the new draft and publish
```

The concept DOI (e.g., `10.5281/zenodo.5654826`) always resolves to the latest
version, while each version gets its own DOI.
