#!/usr/bin/bash

#####################
# Testing constants
#####################
data=avalancheSingleLabel # Note: requires generating updated label files with only label "0" at yolo_single_label_split 
wandb=paper_yolo # wandb project
val=segmentation/val.py # Testing python script
task=test # Task ("train", "val", "test", "speed", "study")
batch=24 # Batch size
testParams="--task $task --iou-thres 0.05 --conf-thres 0.001 --plot_all_ious=True --batch $batch"


for imSize in 448 896 #672
do
    for seed in 50 100 150
    do
        echo "==============================================="
        echo "Original results for size $imSize seed $seed"
        echo "==============================================="
        python3 $val --weights ${wandb}/yolospp_${imSize}px_s$seed/weights/best.pt --img $imSize --data avalanchesplit${seed}.yaml $testParams
    done
done



for imSize in 448 896 #672
do
    for seed in 50 100 150
    do
        echo "==============================================="
        echo "Label results for size $imSize seed $seed"
        echo "==============================================="
        python3 $val --weights ${wandb}/singlelabel_spp_${imSize}px_s$seed/weights/best.pt --img $imSize --data ${data}${seed}.yaml $testParams
    done
done
