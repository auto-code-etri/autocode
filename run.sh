#!/bin/sh

# Note: not same as paper settings
BATCH_SIZE=2
LR=5e-5
EPOCH=10
ACCUMULATION_STEP=2

DROPOUT=0.1
N_ENC_BLOCK=6
N_DEC_BLOCK=6 
HIDDEN_DIM=512
FC_HIDDEN_DIM=2048
NUM_HEAD=8

python3 src/main.py\
    --batch-size=${BATCH_SIZE}\
    --lr=${LR}\
    --epoch=${EPOCH}\
    --gradient-accumulation-step=${ACCUMULATION_STEP}\
    --dropout=${DROPOUT}\
    --n-enc-block=${N_ENC_BLOCK}\
    --n-dec-block=${N_DEC_BLOCK}\
    --hidden=${HIDDEN_DIM}\
    --fc-hidden=${FC_HIDDEN_DIM}\
    --num-head=${NUM_HEAD}\
    --amp\
    --distributed\
    