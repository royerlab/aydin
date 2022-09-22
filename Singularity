BootStrap: docker
# disk limit reached on github CI: can not re-user the docker image
# thus rebuiling from scratch
#From: ghcr.io/truatpasteurdotfr/aydin:master
From: continuumio/miniconda3:master

%post
apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
DEBIAN_FRONTEND=noninteractive apt-get install -y wget bzip2 libglu1-mesa  libgl1-mesa-dri \
          libfontconfig1 libfreetype6 libxcb-xinerama0 \
          libxcb-shape0 libxcb-util1 \
          libdbus-1-3 libxkbcommon-x11-0 libxcb-icccm4 \
          libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
          libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0 pkg-config libhdf5-103 libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*
# https://royerlab.github.io/aydin/getting_started/install.html

conda update --yes conda && \
    conda update -n base -c defaults conda -y && \
    conda update --all -y 
eval "$(/opt/conda/bin/conda shell.bash hook)" && \
    conda create --name aydin_env python=3.9 && \
    conda activate aydin_env && \
    python -m pip install aydin && \
    conda install cudatoolkit

date +"%Y-%m-%d-%H%M" > /last_update

cat <<EOF>/run-aydin.sh
#!/bin/bash
eval "\$(conda shell.bash hook)"
conda activate aydin_env
export LD_LIBRARY_PATH=/.singularity.d/libs:/opt/conda/envs/aydin_env/lib
exec "\$@"
EOF

chmod 755 /run-aydin.sh
%runscript 
/run-aydin.sh "$@"
