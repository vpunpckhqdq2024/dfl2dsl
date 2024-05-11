#!/bin/bash
docker build -t dfl2dsl .
docker rm dfl2dsl || true
docker run -it --name dfl2dsl -v "$(pwd)":/dfl2dsl dfl2dsl /bin/bash
# run the following
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate dfl2dsl
