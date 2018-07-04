#########################################################################
# File Name: train.sh
# Author: Zhen Shen
# mail: spark80231@gmail.com
# Created Time: 2018年05月16日 星期三 15时00分43秒
#########################################################################
#!/bin/bash
GPU_ID=1
BATCH_SIZE=4
WORKER_NUMBER=6
LEARNING_RATE=4e-3
DECAY_STEP=8
SESSION=4

CUDA_VISIBLE_DEVICES=$GPU_ID
python trainval_net.py \
       --dataset pascal_voc --net vgg16 \
       --bs $BATCH_SIZE --nw $WORKER_NUMBER \
       --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
       --epochs 9 \
	   --disp_interval 100 \
       --cuda \
       --s $SESSION \
       --gpu $GPU_ID \
       --descrip CAMSS