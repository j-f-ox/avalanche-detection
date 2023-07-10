#!/usr/bin/bash

#####################
# Testing constants
#####################
data=avalanchesplit
wandb=paper_yolo # wandb project
val=segmentation/val.py # Testing python script
task=test # Task ("train", "val", "test", "speed", "study")
batch=16
testParams="--task $task  --iou-thres 0.05 --conf-thres 0.001 --plot_all_ious=True --batch 12 --batch $batch"

for imSize in 224 448 672 896 1120
do
    for seed in 50 100 150
    do
        echo ""===============================================""
        echo "Results for size $imSize seed $seed"
        echo "==============================================="
        python3 $val --weights ${wandb}/yolospp_${imSize}px_s$seed/weights/best.pt --img $imSize --data ${data}${seed}.yaml $testParams
    done
done