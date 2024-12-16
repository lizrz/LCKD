#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=0
NB_GPU=1

DATA_ROOT=/root/autodl-tmp/lyc/data/cityscapes

DATASET=cityscapes_domain
TASK=14-1
NAME=LGKD
METHOD=LGKD
OPTIONS="--checkpoint checkpoints/step/ --pod local --pod_factor 0.0001 --pod_logits --backbone swin_b"

NB_EPOCHS=50

SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"

RESULTSFILE=results/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"

BATCH_SIZE=16

CUDA_VISIBLE_DEVICES=${GPU} torchrun --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.01 --epochs ${NB_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} torchrun --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 1 --lr 0.001 --epochs ${NB_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.0001, \"type\": \"local\"}}}"
CUDA_VISIBLE_DEVICES=${GPU} torchrun --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 2 --lr 0.001 --epochs ${NB_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.0001, \"type\": \"local\"}}}"
CUDA_VISIBLE_DEVICES=${GPU} torchrun --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 3 --lr 0.001 --epochs ${NB_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.0001, \"type\": \"local\"}}}"
CUDA_VISIBLE_DEVICES=${GPU} torchrun --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 4 --lr 0.001 --epochs ${NB_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.0001, \"type\": \"local\"}}}"
CUDA_VISIBLE_DEVICES=${GPU} torchrun --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 5 --lr 0.001 --epochs ${NB_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.0001, \"type\": \"local\"}}}"
CUDA_VISIBLE_DEVICES=${GPU} torchrun --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"
