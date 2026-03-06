# Model design, forward calls, and possible mismatches (14B vs 5B)

This document describes how the DreamZero VLA is structured, how the forward pass works for both **Wan2.1-I2V-14B** and **Wan2.2-TI2V-5B** backbones, and where mismatches can occur.

---

## 1. Model design

### 1.1 High-level architecture

```
inputs (video, text, actions, state)
        │
        ▼
┌───────────────────┐
│  prepare_input    │  → backbone_inputs, action_inputs
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  backbone         │  → video/text features (e.g. for conditioning)
└─────────┬─────────┘
          │
          ▼
┌───────────────────────────────────────────────────────────────────┐
│  action_head (WANPolicyHead)                                       │
│  ├── encode_prompt, encode_image (CLIP), encode_video (VAE)         │
│  ├── optional: resize video for 5B to target (e.g. 160×320)         │
│  ├── seq_len = num_frames * (H_lat//2) * (W_lat//2)                │
│  └── model (CausalWanModel = DiT + action/state registers)         │
│        → video_noise_pred, action_noise_pred                       │
└───────────────────────────────────────────────────────────────────┘
```

- **Backbone**: Produces conditioning (e.g. text); not the main video denoiser.
- **Action head**: Contains the **DiT** (diffusion transformer), VAE, text/image encoders, and action/state encoders. It computes the video and action noise-prediction loss.
- **DiT (CausalWanModel)**: Same class for both 14B and 5B; implementation: `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`. Architecture and behavior are controlled by config. For **5B (Wan22)** the canonical config is `groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf_wan22.yaml` → `action_head_cfg.config.diffusion_model_cfg` (`dim`, `in_dim`, `frame_seqlen`, `concat_first_frame_latent`, `ffn_dim`, `out_dim`, `num_heads`, `num_layers`, etc.). See `CausalWanModel.__init__` for the full parameter list.

### 1.2 DiT input layout (training, with teacher forcing)

The DiT sees a single sequence built as:

1. **Noisy video tokens** (from noisy latents): length `seq_len = num_frames * tokens_per_frame`.
2. **Action register**: `[action_features, state_features]`, length `action_register_length`.
3. **Teacher forcing**: Prepend **clean** video tokens (same shape as noisy) and duplicate time embeddings, so total length is `2*seq_len + action_register_length`.

So the full sequence is: `[clean_video (seq_len)][noisy_video (seq_len)][action_register (action_register_length)]`.

Block layout (for causal attention and action chunks):

- `noisy_frames = seq_len // frame_seqlen` (must equal `num_frames`).
- `num_image_blocks = (noisy_frames - 1) // num_frame_per_block`.
- `action_horizon = num_image_blocks * num_action_per_block`, `state_horizon = num_image_blocks * num_state_per_block`.
- **Invariant**: `action_register_length == action_horizon + state_horizon`.

Patch embedding uses **stride (1, 2, 2)** on the latent grid, so **tokens_per_frame** (and thus `frame_seqlen` in config) must be **(H_lat//2) * (W_lat//2)**.

---

## 2. 14B vs 5B backbone comparison

| Aspect | Wan2.1-I2V-14B | Wan2.2-TI2V-5B |
|--------|-----------------|-----------------|
| **Config** | `wan_flow_matching_action_tf` (default) | `wan_flow_matching_action_tf_wan22` |
| **Model type** | i2v (image-to-video) | ti2v (text/image-to-video) |
| **VAE** | 16-channel, 8× spatial | WanVideoVAE38, 48-channel, 16× spatial |
| **DiT in_dim** | 36 (latent + first-frame conditioning) | 48 (latent only) |
| **DiT out_dim** | 16 | 48 |
| **concat_first_frame_latent** | **True**: concat `[x; y]` before patch_embedding (first-frame latent in channel dim) | **False**: latent only; first-frame via CLIP in context |
| **First-frame conditioning** | In DiT input (y concatenated to x) | In context (CLIP embedding), not in DiT input |
| **frame_seqlen** | 880 (for 480×256 or similar; patch output per frame) | 50 (160×320 → latent 10×20; or 176×320 → 11×20) |
| **Typical resolution** | e.g. 480×256 (data config) | 160×320 (droid_relative_wan22; even latent, no crop) |
| **Action head resize** | No (14B not in (50,55)) | Yes: wan22 config sets `target_video_height: 160`, `target_video_width: 320` |
| **seq_len formula** | Same: `num_frames * (height//2) * (width//2)` from **noise** (latent) | Same |
| **DiT dim / layers / heads** | 5120 / 40 / 40 | 3072 / 30 / 24 |

So:

- **14B**: I2V, first frame baked into the latent input to the DiT (`concat_first_frame_latent=True`), higher resolution and much larger `frame_seqlen`, no resize in the action head.
- **5B**: TI2V, first frame only in context (CLIP), latent-only DiT input, lower resolution (160×320), small `frame_seqlen` (50), and the action head resizes to the configured target (160×320 in wan22 config).

---

## 3. Forward call flow (unified, with 14B vs 5B branches)

### 3.1 Action head `forward()` (training)

1. **Inputs**: `videos` `[B, C, T, H, W]`, `actions`, `state`, text, etc.
2. **Resize (5B only)**: If config sets `target_video_height` and `target_video_width`, resize to that; else if `getattr(self.model, "frame_seqlen", None) in (50, 55)`, resize to `(176, 320)`. Use **160×320** so latent is 10×20 (even H,W with WanVideoVAE38 16×) and dynamics loss needs no crop. (14B does not hit this.)
3. **Encode**:  
   - `latents = encode_video(videos)` (VAE);  
   - `noise = randn_like(latents)`;  
   - `height, width` = spatial dimensions of **noise** (latent grid).
4. **seq_len**:  
   - `tokens_per_frame = (height // 2) * (width // 2)` (patch output per frame);  
   - `seq_len = num_frames * tokens_per_frame`.  
   Same for 14B and 5B.
5. **Timesteps**: One per frame → `timestep` `[B, num_frames]`; action/state timesteps aligned to blocks.
6. **Call DiT**:  
   `self.model(noisy_latents.transpose(1,2), timestep=..., seq_len=seq_len, state=state_features, action=noisy_actions, timestep_action=..., clean_x=latents.transpose(1,2), context=prompt_embs, clip_feature=..., y=...)`.

### 3.2 DiT `_forward_train()`

1. **x** = noisy latents (and optionally **y** = first-frame latent).
   - **14B**: If `y is not None` and `concat_first_frame_latent`, then `x = cat([x, y], dim=1)` (channel dim) before patch.
   - **5B**: `concat_first_frame_latent` is False → no concat; first-frame is only in context (CLIP).
2. **patch_embedding(x)** with stride (1,2,2) → `x` shape `[B, L, C]` with `L = T * (H_lat//2) * (W_lat//2)`.
3. **Assert** `x.shape[1] == seq_len`.
4. **F** = `timestep.shape[1]` = num_frames. **timestep** expanded to one per token: `expand(B, F, seq_len // F)` → requires `seq_len % F == 0`.
5. **Action branch**: `action_register = [action_features, state_features]`, `x = [x, action_register]` → length `seq_len + action_register_length`.
6. **Teacher forcing**: If `clean_x` is not None, patch-embed clean_x (and for 14B optionally concat y the same way), then `x = [clean_x, x]` → total **s** = `2*seq_len + action_register_length`.
7. **Block layout**:  
   - `half_seq_len = (s - action_register_length) // 2 = seq_len`.  
   - `noisy_frames = half_seq_len // self.frame_seqlen` (must equal `num_frames`).  
   - `num_image_blocks = (noisy_frames - 1) // num_frame_per_block`.  
   - `action_horizon = num_image_blocks * num_action_per_block`, `state_horizon = num_image_blocks * num_state_per_block`.  
   - Check: total length = `half_seq_len + noisy_image_seq_len + action_horizon + state_horizon` ⇒ **action_register_length** must equal **action_horizon + state_horizon**.

### 3.3 Invariants (both backbones)

- `seq_len = num_frames * (H_lat//2) * (W_lat//2)` (action head and DiT patch output).
- `frame_seqlen == (H_lat//2) * (W_lat//2)` for the resolution in use.
- `noisy_frames = seq_len // frame_seqlen = num_frames`.
- `action_register_length = num_image_blocks * (num_action_per_block + num_state_per_block)`.
- `(noisy_frames - 1) // num_frame_per_block >= 1` (at least one image block).

---

## 4. Possible mismatches (and which backbone they affect)

| # | Where | Condition | Failure mode | 14B | 5B |
|---|--------|------------|--------------|-----|-----|
| 1 | Action head → DiT | `seq_len = num_frames * (H_lat//2)*(W_lat//2)` | `x.shape[1] != seq_len` after patch | ✓ | ✓ |
| 2 | Action head | 5B: target resize runs or data matches 160×320 | Wrong latent size → wrong seq_len / layout | — | ✓ |
| 3 | CLI / script | Do **not** override `frame_seqlen` for Wan22 | `noisy_frames` wrong → ValueError block layout | — | ✓ |
| 4 | DiT | `clean_x` same spatial shape as noisy (and same concat for 14B) | `clean_x.shape[1] != seq_len` | ✓ | ✓ |
| 5 | DiT | `seq_len % num_frames == 0` | Wrong timestep expansion | ✓ | ✓ |
| 6 | Config | `frame_seqlen == (H_lat//2)*(W_lat//2)` | `noisy_frames != num_frames`, block layout wrong | ✓ | ✓ (50 for 160×320) |
| 7 | Data/transform | `action_register_length == action_horizon + state_horizon` | ValueError in block layout check | ✓ | ✓ |
| 8 | Data | `(noisy_frames - 1) // num_frame_per_block >= 1` | num_image_blocks = 0 or layout assert | ✓ | ✓ |
| 9 | Backbone/DiT | `context.shape[1] == text_len` | AssertionError | ✓ | ✓ |

**14B-specific**: Uses high `frame_seqlen` (e.g. 880) and consistent resolution; no resize in the action head, so resolution/latent size usually match by data config. Mismatches 2 and 3 do not apply.

**5B-specific**: (2) Resize to target resolution must run (or data must match); wan22 config can set `target_video_height: 160`, `target_video_width: 320` so latent is 10×20 (even). (3) CLI/script must not override `frame_seqlen` to 55. (6) `frame_seqlen` must be 50 (patch output per frame).

### 4.1 Even latent (no crop in dynamics loss)

DiT patch_embedding has stride (1,2,2), so when latent **H** or **W** is odd, unpatchify output has one fewer row/column and the dynamics loss would compare mismatched shapes. Two options:

- **Crop (default)**: Action head crops `training_target` to `video_noise_pred` shape before MSE (no gradient on the missing row/column).
- **Even latent**: Use video resolution so latent H and W are **even**. WanVideoVAE38 has **16×** spatial downscale, so use **video H and W divisible by 32** (e.g. **160×320**). Then latent is 10×20, `(10//2)*(20//2)=50` = frame_seqlen, and no crop is needed. Set `data=dreamzero/droid_relative_wan22` (160×320) and `model/dreamzero/action_head=wan_flow_matching_action_tf_wan22` (which sets `target_video_height: 160`, `target_video_width: 320`).

---

## 5. Summary

- **Design**: VLA = backbone + action head; action head = encoders + CausalWanModel (DiT with action/state registers). Same DiT class for 14B and 5B; behavior differs by config (in_dim, concat_first_frame_latent, frame_seqlen, resolution).
- **Forward**: Action head encodes video, computes `seq_len` from patch layout, optionally resizes (5B to target 160×320), then calls the DiT with noisy/clean latents, timesteps, action/state register, and context. The DiT patch-embeds (with optional 14B [x;y] concat), builds the full sequence with teacher forcing, and checks block layout against `action_register_length`.
- **14B vs 5B**: 14B uses I2V, first-frame in DiT input, high resolution and large `frame_seqlen`; 5B uses TI2V, first-frame in context only, 160×320 and `frame_seqlen=50`, with action head resize to target 160×320. Both use the same seq_len formula and block-layout invariants; 5B is more sensitive to resolution and `frame_seqlen` overrides.

**Current state (verified)**: Wan22 5B smoke test runs successfully. Action head uses patch-output `seq_len`; Wan22 config has `frame_seqlen: 50` and `target_video_height: 160`, `target_video_width: 320` (even latent, no crop in dynamics loss). Data config `dreamzero/droid_relative_wan22` uses 160×320. If latent H or W is odd (e.g. other resolutions), the action head crops `training_target` to `video_noise_pred` shape before dynamics MSE. Scripts do not override `frame_seqlen`; use `model/dreamzero/action_head=wan_flow_matching_action_tf_wan22` and `data=dreamzero/droid_relative_wan22` for 5B DROID training.
