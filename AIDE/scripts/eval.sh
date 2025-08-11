
GPU_NUM=1
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29572

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
PY_ARGS=${@:1}  # Any other arguments
# -m torch.distributed.launch $DISTRIBUTED_ARGS
    # --data_path dataset/Chameleon \
    # --eval_data_path dataset/Chameleon \
python -m torch.distributed.launch $DISTRIBUTED_ARGS main_finetune.py \
    --model AIDE \
    --batch_size 8 \
    --blr 5e-4 \
    --epochs 5 \
    --data_path SDXL \
    --eval_data_path SDXL \
    --resume GenImage_train.pth \
    --eval True \
    --output_dir results
    ${PY_ARGS} 
