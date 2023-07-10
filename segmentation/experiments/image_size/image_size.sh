#!/usr/bin/bash

#####################
# Training constants
#####################
device=0
batch=16
workers=16
epochs=180
patience=25 # Early stopping patience
data=avalanchesplit
wandb=paper_yolo # wandb project
hyp=hyp.scratch_withConf.yaml  # Hyperparameters
train=segmentation/train.py # Train python script
trainParams="--device $device --batch $batch --workers $workers --epochs $epochs --project $wandb --hyp $hyp --patience $patience"
yoloSpp="--weights yolov3-spp.pt $trainParams"


for seed in 50 100 150
do
    for imSize in 224 448 672 896
    do
        python3 $train --name yolospp_${imSize}px_s$seed $yoloSpp --img $imSize --data ${data}${seed}.yaml
    done
done


batch=12
trainParams="--device $device --batch $batch --workers $workers --epochs $epochs --project $wandb --hyp $hyp --patience $patience"
yoloSpp="--weights yolov3-spp.pt $trainParams"
for seed in 50 100 150
do
    for imSize in 1120
    do
        python3 $train --name yolospp_${imSize}px_s$seed $yoloSpp --img $imSize --data ${data}${seed}.yaml
    done
done