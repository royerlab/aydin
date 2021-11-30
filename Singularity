BootStrap: docker
From: ghcr.io/truatpasteurdotfr/aydin:master

%post
date +"%Y-%m-%d-%H%M" > /last_update

%runscript -c /bin/bash
eval "$(conda shell.bash hook)"
conda activate aydin_env
aydin "$@"

