#!/usr/bin/bash

#####################
# Training constants
#####################
device=0
batch=16
workers=16
epochs=180
patience=25 # Early stopping patience
data=avalancheSingleLabel
wandb=paper_yolo # wandb project
hyp=hyp.scratch_withConf.yaml  # Hyperparameters
train=segmentation/train.py # Train python script
trainParams="--device $device --batch $batch --workers $workers --epochs $epochs --project $wandb --hyp $hyp --patience $patience"
yoloSpp="--weights yolov3-spp.pt $trainParams"


for imSize in 448 672 896
do
    for seed in 50 100 150
    do
        python3 $train --name singlelabel_spp_${imSize}px_s$seed $yoloSpp --img $imSize --data ${data}${seed}.yaml
    done
done