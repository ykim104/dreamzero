#!/bin/bash
# DreamZero MjThor Full Fine-Tuning Script with Wan2.1-I2V-14B-480P backbone
#
# Usage:
#   bash scripts/train/mjthor_training_full_finetune_wan21_14b.sh
#
# Prerequisites:
#   - MjThor dataset at MJTHOR_DATA_ROOT (default: /weka/prior/datasets/robomolmo/feb10_franka_and_rby1/FrankaPickOmniCamConfig/train)
#   - Wan2.1-I2V-14B-480P weights (auto-downloaded or pre-downloaded from HuggingFace)
#     huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
#   - umt5-xxl tokenizer (auto-downloaded or pre-downloaded)
#     huggingface-cli download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl
#
# DataLoader: dataloader_num_workers=0 and smaller num_steps_per_shard avoid OOM
# on Beaker (workers were killed when caching full shards in parallel).

export HYDRA_FULL_ERROR=1

# Repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -n "$DREAMZERO_ROOT" ] && [ -d "$DREAMZERO_ROOT/groot" ]; then
    :
elif [ -d "/root/yejink/dreamzero/groot" ]; then
    DREAMZERO_ROOT=/root/yejink/dreamzero
elif [ -d "/root/dreamzero/groot" ]; then
    DREAMZERO_ROOT=/root/dreamzero
elif [ -d "$SCRIPT_REPO_ROOT/groot" ]; then
    DREAMZERO_ROOT="$SCRIPT_REPO_ROOT"
else
    DREAMZERO_ROOT="${DREAMZERO_ROOT:-/root/yejink/dreamzero}"
fi
if [ ! -d "$DREAMZERO_ROOT/groot" ]; then
    echo "ERROR: No groot/ under $DREAMZERO_ROOT. Set DREAMZERO_ROOT to the dreamzero repo root that contains groot/."
    exit 1
fi

# ============ USER CONFIGURATION ============
MJTHOR_DATA_ROOT=${MJTHOR_DATA_ROOT:-"/weka/prior/datasets/robomolmo/feb10_franka_and_rby1/FrankaPickOmniCamConfig/train"}
OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_mjthor_wan21_14b_full_finetune"}

NUM_GPUS=${NUM_GPUS:-1}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_BS))}

# Wan2.1-I2V-14B-480P checkpoint (14B model; includes VAE and CLIP)
WAN14B_CKPT_DIR=${WAN14B_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$DREAMZERO_ROOT/checkpoints/umt5-xxl"}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN14B_CKPT_DIR" ] || [ -z "$(ls -A "$WAN14B_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN14B_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN14B_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
# ================================================

if [ ! -d "$MJTHOR_DATA_ROOT" ]; then
    echo "ERROR: MjThor dataset not found at $MJTHOR_DATA_ROOT"
    exit 1
fi

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"
if [ ! -f "$EXPERIMENT_PY" ]; then
    echo "ERROR: Not found: $EXPERIMENT_PY"
    exit 1
fi
PYTHON_311="/usr/bin/python3.11"
if [ -x "$PYTHON_311" ]; then
    if [ -n "${FIX_NUMPY_IN_SCRIPT:-}" ]; then
        "$PYTHON_311" -m pip install "numpy==1.26.4" --force-reinstall -q 2>/dev/null || true
    fi
    RUN_CMD=( "$PYTHON_311" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" )
    echo "Using image Python 3.11: $PYTHON_311"
else
    RUN_CMD=( python3 -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" )
    echo "Using: $(command -v python3)"
fi
cd "$DREAMZERO_ROOT"

DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2_offload}
DEEPSPEED_CFG_PATH="$DREAMZERO_ROOT/groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json"
if [ ! -f "$DEEPSPEED_CFG_PATH" ]; then
    echo "ERROR: DeepSpeed config not found at $DEEPSPEED_CFG_PATH"
    exit 1
fi

# 14B model: wan_flow_matching_action_tf (no _wan22), num_frame_per_block=1, num_action_per_block=32, frame_seqlen=880, image height 176
"${RUN_CMD[@]}" \
    report_to=wandb \
    data=dreamzero/mjthor_relative \
    wandb_project=dreamzero \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=1 \
    num_action_per_block=32 \
    num_state_per_block=1 \
    frame_seqlen=880 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="$DEEPSPEED_CFG_PATH" \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$PER_DEVICE_BS \
    global_batch_size=$GLOBAL_BATCH_SIZE \
    max_steps=100000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=true \
    dataloader_num_workers=0 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=false \
    max_chunk_size=4 \
    save_strategy=steps \
    train_dataset.num_steps_per_shard=5000 \
    mjthor_data_root=$MJTHOR_DATA_ROOT \
    dit_version=$WAN14B_CKPT_DIR \
    text_encoder_pretrained_path=$WAN14B_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN14B_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN14B_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
