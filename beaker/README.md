# Beaker: DreamZero Wan2.2 training

## Build the image

From the **repo root** (parent of `beaker/`). The `-t` flag tags the image; without it the image will show as `<none>`.

```bash
cd /path/to/dreamzero
docker build -f beaker/Dockerfile -t dreamzero-wan22:latest .
```

You should see `Successfully tagged dreamzero-wan22:latest` at the end. Then push to Beaker:

```bash
beaker image create --name yejink/dreamzero-wan22 dreamzero-wan22:latest
```

## Test the image locally

Quick sanity check that PyTorch, CUDA, `flash_attn`, `decord`, and `librosa` all import and see GPUs:

To still test the image CPU‑only, just drop the --gpus all flag:
```bash
docker run --rm --gpus all \
  -v $PWD:/workspace/dreamzero \
  dreamzero-wan22:latest \
  bash -lc "python - << 'EOF'
import torch
from flash_attn import flash_attn_interface  # ensure flash_attn is usable
import decord, librosa

print('Torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
print('flash_attn OK, decord OK, librosa OK')
EOF"
```

If that finishes without errors and reports CUDA available, the image is ready. If you see `ArgumentError: activate does not accept more than one argument`, the base image has an entrypoint; use `--entrypoint ""`:

```bash
docker run --rm --entrypoint "" --gpus all \
  -v $PWD:/workspace/dreamzero \
  dreamzero-wan22:latest \
  bash -c "python3.11 -c \"
import torch
from flash_attn import flash_attn_interface
import decord, librosa
print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'Device count:', torch.cuda.device_count())
\""
```

## Create the Beaker image

From the same directory where you built the image:

```bash
beaker image create --name dreamzero-wan22 dreamzero-wan22:latest
```

Then note the image name/ID for use in the experiment spec.

## Run the experiment

1. In `beaker/experiment.yaml`, set `tasks[0].image.beaker` to your image name or Beaker image ID.
2. (Optional) Uncomment and set `datasets` if you use Beaker datasets for DROID data or checkpoints.
3. Create the experiment:

```bash
beaker experiment create beaker/experiment.yaml
```

The experiment runs 4-GPU training via `scripts/train/droid_training_wan22.sh`.
