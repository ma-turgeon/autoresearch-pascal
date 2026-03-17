#!/bin/bash
# run.sh — Run a single training experiment
# Usage: ./run.sh
# Must have conda env 'autoresearch' active and CUDA_VISIBLE_DEVICES set
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
python train.py
