#!/bin/bash
# Smoke test: run 2 training steps to verify the training script and env.
# Usage: bash scripts/train/droid_training_wan22_smoke_test.sh
# Requires: DROID_DATA_ROOT to exist and have at least one episode (partial download OK).

export HYDRA_FULL_ERROR=1

# Smoke-test settings
NUM_GPUS=${NUM_GPUS:-1}
DROID_DATA_ROOT=${DROID_DATA_ROOT:-"./data/droid_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_droid_wan22_smoke"}
WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"./checkpoints/Wan2.2-TI2V-5B"}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}

# Same auto-download as main script
if [ ! -d "$WAN22_CKPT_DIR" ] || [ -z "$(ls -A "$WAN22_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.2-TI2V-5B not found. Downloading..."
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN22_CKPT_DIR"
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl not found. Downloading..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
if [ ! -f "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "Image encoder not found. Downloading Wan2.1 (CLIP)..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$IMAGE_ENCODER_DIR"
fi

if [ ! -d "$DROID_DATA_ROOT" ]; then
    echo "ERROR: DROID dataset not found at $DROID_DATA_ROOT"
    exit 1
fi

# Ensure hydra is available (e.g. when container uses a different Python than the one used at build)
pip install hydra-core --quiet

echo "Smoke test: max_steps=2, NUM_GPUS=$NUM_GPUS, output=$OUTPUT_DIR"

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/droid_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=2 \
    weight_decay=1e-5 \
    save_total_limit=1 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=0 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=55 \
    save_strategy=no \
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
