#!/usr/bin/bash

#####################
# Testing constants
#####################
patience=25 # Early stopping patience
data=avalanchesplit
wandb=paper_yolo # wandb project
val=segmentation/val.py # Testing python script
task=val # Task ("train", "val", "test", "speed", "study")
batch=16 # Batch size
testParams="--task $task --iou-thres 0.05 --conf-thres 0.001 --plot_all_ious=True --batch $batch"


for imSize in 448 896
do
    for seed in 50 100 150
    do
        echo "==============================================="
        echo "Results for norm size $imSize seed $seed"
        echo "==============================================="
        python3 $val --weights ${wandb}/yolo${imSize}_norm_p${patience}s$seed/weights/best.pt --img $imSize --data ${data}${seed}.yaml $testParams
    done
    
    for seed in 50 100 150
    do
        echo "==============================================="
        echo "Results for spp size $imSize seed $seed"
        echo "==============================================="
        python3 $val --weights ${wandb}/yolo${imSize}_spp_p${patience}s$seed/weights/best.pt --img $imSize --data ${data}${seed}.yaml $testParams
    done
    
    for seed in 50 100 150
    do
        echo "==============================================="
        echo "Results for tiny size $imSize seed $seed"
        echo "==============================================="
        python3 $val --weights ${wandb}/yolo${imSize}_tiny_p${patience}s$seed/weights/best.pt --img $imSize --data ${data}${seed}.yaml $testParams
    done
done