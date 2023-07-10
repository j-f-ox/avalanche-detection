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
yoloNorm="--weights yolov3.pt $trainParams"
yoloSpp="--weights yolov3-spp.pt $trainParams"
yoloTiny="--weights yolov3-tiny.pt $trainParams"


# Experiments with image size 448px
imSize=448
for seed in 50 100 150
do
    python3 $train --name yolo${imSize}_norm_p${patience}s$seed $yoloNorm --img $imSize --data ${data}${seed}.yaml
    python3 $train --name yolo${imSize}_spp_p${patience}s$seed $yoloSpp --img $imSize --data ${data}${seed}.yaml
    python3 $train --name yolo${imSize}_tiny_p${patience}s$seed $yoloTiny --img $imSize --data ${data}${seed}.yaml
done

# Experiments with image size 896px
imSize=896
for seed in 50 100 150
do
    python3 $train --name yolo${imSize}_norm_p${patience}s$seed $yoloNorm --img $imSize --data ${data}${seed}.yaml
    python3 $train --name yolo${imSize}_spp_p${patience}s$seed $yoloSpp --img $imSize --data ${data}${seed}.yaml
    python3 $train --name yolo${imSize}_tiny_p${patience}s$seed $yoloTiny --img $imSize --data ${data}${seed}.yaml
done

