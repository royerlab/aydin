BootStrap: docker
From: ghcr.io/truatpasteurdotfr/aydin:master

%post
date +"%Y-%m-%d-%H%M" > /last_update

%runscript 
eval "$(conda shell.bash hook)"
conda activate aydin_env
export LD_LIBRARY_PATH=/.singularity.d/libs:/opt/conda/envs/aydin_env/lib
eval "$@"

