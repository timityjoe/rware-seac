#!/bin/bash

echo "Setting up RWARE SEAC Environment..."
# source activate base	
# conda deactivate
conda activate conda38-rware-seac
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
echo "$LD_LIBRARY_PATH"
