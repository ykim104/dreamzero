"""
Serve a DreamZero policy with Wan2.2-TI2V-5B backbone over the websocket policy server.

The checkpoint at model_path must have been trained with the Wan22 config
(model/dreamzero/action_head=wan_flow_matching_action_tf_wan22) so that
experiment_cfg/conf.yaml and the saved weights use the 5B DiT and VAE38.

Usage (single GPU):

  # Option A: with torchrun (recommended)
  torchrun --nproc_per_node=1 eval_utils/serve_dreamzero_wan22.py --model_path ./checkpoints/dreamzero_droid_wan22_smoke --port 8000

  # Option B: single process (initializes dist with world_size=1)
  python eval_utils/serve_dreamzero_wan22.py --model_path ./checkpoints/dreamzero_droid_wan22_smoke --port 8000

  # When loading from a checkpoint trained with a local tokenizer path, either ensure that path
  # exists, or override it: --tokenizer_path google/umt5-xxl (downloads) or --tokenizer_path /path/to/umt5-xxl

Client should send observations in the format expected by PolicyServerConfig (see policy_server.py).
The model resizes video to 160×320 internally (Wan22 config target_video_height/width), so the client
can send other resolutions; using (160, 320) in server_config is recommended to reduce bandwidth.
"""

import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import tyro
from pathlib import Path
from tianshou.data import Batch

# Add repo root for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openpi_client.base_policy import BasePolicy

from eval_utils.policy_server import WebsocketPolicyServer, PolicyServerConfig
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag


# Wan22 uses 160x320 (H, W)
WAN22_IMAGE_HEIGHT = 160
WAN22_IMAGE_WIDTH = 320
FRAMES_PER_CHUNK = 4


def _maybe_init_distributed():
    """Initialize process group for single-GPU or multi-GPU. Required by GrootSimPolicy."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)


class DreamZeroWan225BPolicy(BasePolicy):
    """
    Wraps GrootSimPolicy (DreamZero with Wan2.2-TI2V-5B) for the websocket policy server.

    Converts between roboarena observation/action format and the Batch format
    expected by GrootSimPolicy (DROID modality keys). The action head resizes
    video to 160×320 internally for Wan22; server_config can hint (160, 320) to the client.
    """

    def __init__(self, groot_policy: GrootSimPolicy):
        super().__init__()
        self._policy = groot_policy
        self._frame_buffers = {
            "video.exterior_image_1_left": [],
            "video.exterior_image_2_left": [],
            "video.wrist_image_left": [],
        }
        self._is_first_call = True
        self._current_session_id = None

    def _convert_observation(self, obs: dict) -> dict:
        """Convert roboarena observation format to DROID/Batch format."""
        image_key_mapping = {
            "observation/exterior_image_0_left": "video.exterior_image_1_left",
            "observation/exterior_image_1_left": "video.exterior_image_2_left",
            "observation/wrist_image_left": "video.wrist_image_left",
        }
        for roboarena_key, droid_key in image_key_mapping.items():
            if roboarena_key in obs:
                data = obs[roboarena_key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        self._frame_buffers[droid_key].extend(list(data))
                    else:
                        self._frame_buffers[droid_key].append(data)

        num_frames = 1 if self._is_first_call else FRAMES_PER_CHUNK
        converted = {}
        for droid_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    frames_to_use = buffer[-num_frames:]
                else:
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        frames_to_use.insert(0, buffer[0])
                video = np.stack(frames_to_use, axis=0)
                converted[droid_key] = video

        if "observation/joint_position" in obs:
            joint_pos = np.asarray(obs["observation/joint_position"])
            if joint_pos.ndim == 1:
                joint_pos = joint_pos.reshape(1, -1)
            converted["state.joint_position"] = joint_pos.astype(np.float64)
        else:
            converted["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)

        if "observation/gripper_position" in obs:
            gripper_pos = np.asarray(obs["observation/gripper_position"])
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos.reshape(1, -1)
            converted["state.gripper_position"] = gripper_pos.astype(np.float64)
        else:
            converted["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)

        converted["annotation.language.action_text"] = obs.get("prompt", "")
        return converted

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """Convert model action dict to (N, 8) array (7 joint + 1 gripper)."""
        joint_action = None
        gripper_action = None
        for key, value in action_dict.items():
            if "joint_position" in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value
        if joint_action is None:
            return np.zeros((1, 8), dtype=np.float32)
        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)
        N = joint_action.shape[0]
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)
        return np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)

    def infer(self, obs: dict) -> np.ndarray:
        session_id = obs.get("session_id")
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                self.reset({})
            self._current_session_id = session_id

        converted_obs = self._convert_observation(obs)
        batch = Batch(obs=converted_obs)
        with torch.no_grad():
            result_batch, _ = self._policy.lazy_joint_forward_causal(batch)
        action_dict = {}
        action_chunk_dict = result_batch.act
        for k in dir(action_chunk_dict):
            if k.startswith("action."):
                action_dict[k] = getattr(action_chunk_dict, k)
        action = self._convert_action(action_dict)
        if self._is_first_call:
            self._is_first_call = False
        return action

    def reset(self, reset_info: dict) -> None:
        for key in self._frame_buffers:
            self._frame_buffers[key] = []
        self._is_first_call = True
        self._current_session_id = None
        if hasattr(self._policy.trained_model, "action_head") and hasattr(
            self._policy.trained_model.action_head, "current_start_frame"
        ):
            self._policy.trained_model.action_head.current_start_frame = 0


def main(
    model_path: str = "./checkpoints/dreamzero_droid_wan22_smoke",
    tokenizer_path: str | None = None,
    port: int = 8000,
    host: str = "0.0.0.0",
) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    logger = logging.getLogger(__name__)

    _maybe_init_distributed()
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))

    logger.info("Loading DreamZero Wan22 policy from %s", model_path)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("oxe_droid"),
        model_path=model_path,
        tokenizer_path_override=tokenizer_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )
    wrapper = DreamZeroWan225BPolicy(groot_policy=policy)

    server_config = PolicyServerConfig(
        image_resolution=(WAN22_IMAGE_HEIGHT, WAN22_IMAGE_WIDTH),
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="joint_position",
    )
    logger.info("Starting WebsocketPolicyServer on %s:%d (Wan22 160x320)", host, port)
    server = WebsocketPolicyServer(
        policy=wrapper,
        server_config=server_config,
        host=host,
        port=port,
    )
    server.serve_forever()


if __name__ == "__main__":
    tyro.cli(main)
