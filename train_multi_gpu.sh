#!/bin/bash
# 多GPU并行训练5个任务

# 任务配置 - 使用GPU 0和1轮流分配
TASKS=(
  "close_upper_drawer:/data/wxn/zarr_data/close_the_upper_drawer:6"
  "flip_cup:/data/wxn/zarr_data/flip_cup:6" 
  "put_bottle_on_plate:/data/wxn/zarr_data/put_the_bottle_on_the_plate:7"
  "put_gorilla_on_shelf:/data/wxn/zarr_data/put_the_gorilla_on_the_top_shelf:7"
  "put_lion_on_shelf:/data/wxn/zarr_data/put_the_lion_on_the_top_shelf:7"
)

# 并行启动训练
for task_info in "${TASKS[@]}"; do
  IFS=":" read -r task_name dataset_path gpu_id <<< "$task_info"
  
  echo "Starting training for $task_name on GPU $gpu_id"
  
  # 后台启动训练
  CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
    --config-name=train_my_real_image_workspace \
    task.name=$task_name \
    task.dataset_path=$dataset_path \
    training.device=cuda:0 \
    dataloader.batch_size=128 \
    > logs/train_${task_name}.log 2>&1 &
    
  # GPU分配延迟：同一GPU上的任务间隔更长
  if [ "$gpu_id" == "0" ]; then
    sleep 60  # GPU 0的任务启动间隔
  else
    sleep 30  # GPU 1的任务启动间隔
  fi
done

echo "All training jobs started! Check logs/ directory for progress."
echo "Use 'nvidia-smi' to monitor GPU usage."

# 等待所有后台任务完成
wait
echo "All training completed!"
