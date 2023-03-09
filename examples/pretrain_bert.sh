#!/bin/bash

cat bert_text_sentence.bin.* > bert_text_sentence.bin

DATA_PATH=bert_text_sentence
CMD="python -u pretrain_bert.py \
     --use-new-schedule \
     --num-workers 8 \
     --activations-checkpoint-method uniform \
     --activations-checkpoint-num-layers 1 \
     --pipeline-model-parallel-size 4 \
     --num-layers-per-virtual-pipeline-stage 2 \
     --tensor-model-parallel-size 4 \
     --num-layers 32 \
     --hidden-size 4096 \
     --num-attention-heads 32 \
     --micro-batch-size 8 \
     --global-batch-size 128 \
     --seq-length 1024 \
     --max-position-embeddings 1024 \
     --train-iters 1000000 \
     --data-path $DATA_PATH \
     --vocab-file vocab.txt \
     --data-impl mmap \
     --split 949,50,1 \
     --distributed-backend nccl \
     --lr 0.0001 \
     --lr-decay-style linear \
     --min-lr 1.0e-5 \
     --lr-decay-iters 990000 \
     --weight-decay 1e-2 \
     --clip-grad 1.0 \
     --lr-warmup-fraction .01 \
     --log-interval 10 \
     --exit-interval 40 \
     --save-interval 10000 \
     --eval-interval 1000 \
     --eval-iters 10 \
     --fp16"

THIS_DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
LOG_DIR=${THIS_DIR}/logs/$DATETIME
mkdir -p ${LOG_DIR}
cp $0 ${LOG_DIR}/run.sh
echo Commit-Hash `git rev-parse --short HEAD` >> ${LOG_DIR}/version.log

srun --output=${LOG_DIR}/%j.%t.log \
     -p INTERN -N1 -n8 --ntasks-per-node=8 --gres=gpu:8 \
     --kill-on-bad-exit=1 \
     sh -c "${CMD}"