#########################################################################
# File Name: train.sh
# Author: Zhen Shen
# mail: spark80231@gmail.com
# Created Time: 2018年05月16日 星期三 15时00分43秒
#########################################################################
#!/bin/bash
GPU_ID=0
BATCH_SIZE=1
WORKER_NUMBER=6
LEARNING_RATE=1e-4
DECAY_STEP=5

CUDA_VISIBLE_DEVICES=$GPU_ID
python trainval_net.py \
       --dataset fashion_ai --net vgg16 \
       --bs $BATCH_SIZE --nw $WORKER_NUMBER \
       --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
       --epochs 8 \
	   --disp_interval 1 \
       --cuda
