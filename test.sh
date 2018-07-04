#########################################################################
# File Name: test.sh
# Author: Zhen Shen
# mail: spark80231@gmail.com
# Created Time: 2018年05月17日 星期四 12时54分14秒
#########################################################################
#!/bin/bash
SESSION=4
EPOCH=9
CHECKPOINT=2504

python test_net.py \
	   --dataset pascal_voc --net vgg16 \
	   --checksession $SESSION \
	   --checkepoch $EPOCH \
	   --checkpoint $CHECKPOINT \
       --cuda

