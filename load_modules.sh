#!/bin/bash
cd /gpfs/wolf/gen170/proj-shared/FSB/
# unload IBM XL compilers (C/C++)
module unload xl

# load GNU compiler toolchain
module load gcc

# load the OpenCE environment (loads CUDA as well)
module load open-ce/1.1.3-py38-0

cd apex
CC=gcc CXX=g++ python setup.py install --cpp_ext --cuda_ext --prefix $HOME/.local/

bsub -P GEN170 -nnodes 1 -W 90 -Is /bin/bash
