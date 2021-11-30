FROM continuumio/miniconda3:master
MAINTAINER Tru Huynh <tru@pasteur.fr>

RUN apt-get update && \
    apt-get install -y && \
    apt-get install -y wget bzip2 && \
    apt-get install -y libglu1-mesa  libgl1-mesa-dri 

# https://royerlab.github.io/aydin/getting_started/install.html
RUN apt-get update && apt-get install -y \
          libfontconfig1 libfreetype6 libxcb-xinerama0 \
          libxcb-shape0 libxcb-util1 \
          libdbus-1-3 libxkbcommon-x11-0 libxcb-icccm4 \
          libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
          libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0 pkg-config libhdf5-103 libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

RUN conda update --yes conda && \
    conda update -n base -c defaults conda -y && \
    conda update --all -y 
RUN eval "$(/opt/conda/bin/conda shell.bash hook)" && \
    conda create --name aydin_env python=3.9 && \
    conda activate aydin_env && \
    python -m pip install aydin && \
    conda install cudatoolkit<=11.2

RUN date +"%Y-%m-%d-%H%M" > /last_update
