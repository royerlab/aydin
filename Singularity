BootStrap: docker
From: ghcr.io/truatpasteurdotfr/aydin:master

%post
date +"%Y-%m-%d-%H%M" > /last_update

cat <<EOF>/run-aydin.sh
#!/bin/bash
eval "\$(conda shell.bash hook)"
conda activate aydin_env
export LD_LIBRARY_PATH=/.singularity.d/libs:/opt/conda/envs/aydin_env/lib
eval "\$@"
EOF

chmod 755 /run-aydin.sh
%runscript 
/run-aydin.sh
