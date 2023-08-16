#!/bin/sh

# Note: not same as paper settings
BATCH_SIZE=32
LR=5e-5
EPOCH=10
ACCUMULATION_STEP=2

DROPOUT=0.1
N_ENC_BLOCK=12
N_DEC_BLOCK=12
HIDDEN_DIM=768
FC_HIDDEN_DIM=2048
NUM_HEAD=12

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
    --do_ast\
    --distributed\
    