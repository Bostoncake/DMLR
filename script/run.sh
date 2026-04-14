#!/bin/bash
set -euo pipefail

export HF_HOME=~/.huggingface_cache
export CUDA_VISIBLE_DEVICES=7

NUM_WORKERS=1
DATA_POINT=300


MODEL_NAME=/WillDevExt/xiongyizhe/models/Qwen3-VL-4B-Instruct
# MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
# OUTPUT_DIR=./output/v1/${MODEL_NAME}/${DATASET}
for DATASET in math_vista math_vision mm_math hallusion mmvp mmstar scienceqa; do
    # DATASET=mmvp
    DATASET_FILE=mllm_data/${DATASET}_dev.json
    OUTPUT_DIR=./output_vis/0414_reproduce/${DATASET}/Qwen3-VL-4B-Instruct


    python main.py \
        --dataset $DATASET_FILE \
        --model_name_or_path $MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --device cuda \
        --seed 42 \
        --max_new_tokens 2048 \
        --max_num_steps 15 \
        --num_thought_tokens 4 \
        --sigma 0.1 \
        --sigma_decay 0.95 \
        --lr 0.001 \
        --verbose 0 \
        --min_pixels 128 \
        --max_pixels 256 \
        --start_data_idx 0 \
        --end_data_idx $DATA_POINT \
        --use_llm_verify \
        --num_workers $NUM_WORKERS \
        --worker_device_round_robin \
        --num_selected_patches 16 \
        --initial_patch_count 2 \
        --patch_increment 2 \
        --visual_insert_stride 1 \
        --visual_injection_start_step 0 \
        --visual_injection_interval 1 

done