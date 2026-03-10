"""MjThor dataset loader for DreamZero training.

Reads the HDF5+video format used by sim_pick_place / MjThor and exposes the
same shard-based interface consumed by ShardedLeRobotMixtureDataset so the
existing training loop can iterate over it unchanged.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import h5py
import torch.distributed as dist
from decord import VideoReader
from torch.utils.data import IterableDataset, get_worker_info

from groot.vla.data.dataset.lerobot import ModalityConfig
from groot.vla.data.schema import (
    DatasetMetadata,
    DatasetModalities,
    DatasetStatisticalValues,
    DatasetStatistics,
    EmbodimentTag,
    StateActionMetadata,
    VideoMetadata,
)
from groot.vla.data.transform import ComposedModalityTransform


class MjThorSingleDataset:
    """Reads one MjThor data directory (HDF5 trajectories + mp4 videos).

    The public interface mirrors the subset of ShardedLeRobotSingleDataset
    that ShardedLeRobotMixtureDataset.__iter__ and the config system need.

    When ``max_chunk_size > 0``, multi-chunk loading is enabled (matching
    the DROID ``ShardedLeRobotSubLangSingleActionChunkDatasetDROID``).
    Each sample then contains up to ``max_chunk_size`` action-chunks worth
    of video / action / state, expanding outward from the sampled anchor
    in ±24-step increments within the same trajectory.
    """

    def __init__(
        self,
        data_root: str | Path,
        selected_states: list[str],
        selected_actions: list[str],
        selected_observations: list[str],
        embodiment_tag: str | EmbodimentTag = EmbodimentTag.MJTHOR,
        modality_configs: dict[str, ModalityConfig] | None = None,
        transforms: ComposedModalityTransform | None = None,
        house_idxs: list[int] | None = None,
        drop_n_last_frames: int = 0,
        num_steps_per_shard: int = 10_000,
        img_size: tuple[int, int] | None = None,
        max_chunk_size: int = 0,
        relative_action: bool = False,
        relative_action_keys: list[str] | None = None,
    ):
        self.data_root = Path(data_root)
        self.selected_states = selected_states
        self.selected_actions = selected_actions
        self.selected_observations = selected_observations
        self.tag = (
            EmbodimentTag(embodiment_tag)
            if isinstance(embodiment_tag, str)
            else embodiment_tag
        )
        self.modality_configs = modality_configs or {}
        self.transforms = (
            transforms
            if transforms is not None
            else ComposedModalityTransform(transforms=[])
        )
        self.drop_n_last_frames = drop_n_last_frames
        self.num_steps_per_shard = num_steps_per_shard
        self.img_size = img_size
        self.house_idxs = house_idxs
        self.max_chunk_size = max_chunk_size
        self.relative_action = relative_action
        self.relative_action_keys = relative_action_keys or []

        # Bookkeeping
        self.traj_files: list[str] = []
        self.traj_file_traj_idxs: list[int] = []
        self.traj_lengths: list[int] = []
        self._build_bookkeeping()
        self._trajectory_lengths = np.array(self.traj_lengths)
        self._trajectory_ids = np.arange(len(self.traj_lengths))

        self.fps = self._get_fps()
        self._native_video_resolution = self._probe_video_resolution()

        # Modality keys
        self._modality_keys = self._build_modality_keys()
        self._delta_indices = self._build_delta_indices()
        self._max_delta_index = (
            max(int(d.max()) for d in self._delta_indices.values())
            if self._delta_indices
            else 0
        )
        self._step_filter = self._build_step_filter()

        # Statistics and metadata
        self._stats = self._load_stats()
        self._metadata = self._build_metadata()

        # Sharding
        self.sharded_trajectories, self.shard_lengths = self._generate_shards()

        # Shard caching state
        self.shard_start_indices: dict[int, int] | None = None
        self.cached_video_frames: dict[int, dict[str, np.ndarray]] | None = None
        self.cached_states: dict[int, dict[str, np.ndarray]] | None = None
        self.cached_actions: dict[int, dict[str, np.ndarray]] | None = None
        self.cached_languages: dict[int, str] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._cache_job: Future | None = None

        self.set_transforms_metadata(self._metadata)
        print(
            f"MjThorSingleDataset: {len(self)} steps, "
            f"{self.num_shards} shards, fps={self.fps}"
        )

    # -- Properties --

    @property
    def trajectory_ids(self) -> np.ndarray:
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        return self._trajectory_lengths

    @property
    def modality_keys(self) -> dict[str, list[str]]:
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        return self._delta_indices

    @property
    def max_delta_index(self) -> int:
        return self._max_delta_index

    @property
    def step_filter(self) -> dict[int, np.ndarray]:
        return self._step_filter

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    @property
    def num_shards(self) -> int:
        return len(self.sharded_trajectories)

    def __len__(self) -> int:
        return int(self._trajectory_lengths.sum())

    def get_trajectory_index(self, trajectory_id: int) -> int:
        return trajectory_id

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        self.transforms.set_metadata(metadata)

    # -- Bookkeeping --

    def _build_bookkeeping(self):
        index_path = self.data_root / "valid_trajectory_index.json"
        with open(index_path, "r") as f:
            valid_traj_index: dict = json.load(f)

        if self.house_idxs is None:
            self.house_idxs = sorted(
                int(name.split("_")[1]) for name in valid_traj_index.keys()
            )

        for house_idx in self.house_idxs:
            house_key = f"house_{house_idx}"
            if house_key not in valid_traj_index:
                continue
            for datafile_subpath, traj_lens in valid_traj_index[house_key].items():
                for traj_key, traj_len in traj_lens.items():
                    traj_idx = int(traj_key.split("_")[1])
                    traj_len = max(0, traj_len - 2 - self.drop_n_last_frames)
                    if traj_len > 0:
                        self.traj_files.append(datafile_subpath)
                        self.traj_file_traj_idxs.append(traj_idx)
                        self.traj_lengths.append(traj_len)

    def _get_fps(self) -> int:
        if len(self.traj_files) == 0:
            return 10
        file_traj_idx = self.traj_file_traj_idxs[0]
        with h5py.File(self.data_root / self.traj_files[0], "r") as f:
            obs_scene = json.loads(
                f[f"traj_{file_traj_idx}/obs_scene"][()]
                .decode("utf-8")
                .rstrip("\x00")
            )
            fps = round(1000 / obs_scene["policy_dt_ms"])
        return fps

    def _probe_video_resolution(self) -> tuple[int, int]:
        """Read one frame from the first trajectory to get native (W, H)."""
        if len(self.traj_files) == 0 or not self.selected_observations:
            return (640, 360)
        cam = self.selected_observations[0]
        file_traj_idx = self.traj_file_traj_idxs[0]
        h5_path = self.data_root / self.traj_files[0]
        try:
            with h5py.File(h5_path, "r") as f:
                obs_ds = f[f"traj_{file_traj_idx}/obs/sensor_data/{cam}"]
                vname = obs_ds[:].tobytes().decode("utf-8").rstrip("\x00")
            vr = VideoReader(str(h5_path.with_name(vname)))
            frame = vr[0]
            if hasattr(frame, "asnumpy"):
                frame = frame.asnumpy()
            elif hasattr(frame, "numpy"):
                frame = frame.numpy()
            h, w = frame.shape[0], frame.shape[1]
            return (w, h)
        except Exception:
            return (640, 360)

    def _build_modality_keys(self) -> dict[str, list[str]]:
        keys: dict[str, list[str]] = {}
        if self.selected_observations:
            keys["video"] = [f"video.{cam}" for cam in self.selected_observations]
        state_keys = []
        for s in self.selected_states:
            top, sub = s.split(".")
            state_keys.append(f"state.{top}_{sub}")
        if state_keys:
            keys["state"] = state_keys
        action_keys = []
        for a in self.selected_actions:
            top, sub = a.split(".")
            action_keys.append(f"action.{top}_{sub}")
        if action_keys:
            keys["action"] = action_keys
        keys["language"] = ["annotation.language.task_description"]
        return keys

    def _build_delta_indices(self) -> dict[str, np.ndarray]:
        indices: dict[str, np.ndarray] = {}
        for modality, cfg in self.modality_configs.items():
            if isinstance(cfg, dict):
                cfg = ModalityConfig(**cfg)
            for key in cfg.modality_keys:
                indices[key] = np.array(cfg.delta_indices)
        # Fallback defaults when no modality_configs provided
        if not indices:
            for key in self._modality_keys.get("video", []):
                indices[key] = np.arange(25)
            for key in self._modality_keys.get("state", []):
                indices[key] = np.array([0])
            for key in self._modality_keys.get("action", []):
                indices[key] = np.arange(24)
            for key in self._modality_keys.get("language", []):
                indices[key] = np.array([0])
        return indices

    def _build_step_filter(self) -> dict[int, np.ndarray]:
        sf: dict[int, np.ndarray] = {}
        for traj_id in range(len(self.traj_lengths)):
            sf[traj_id] = np.arange(self.traj_lengths[traj_id])
        return sf

    # -- Statistics and metadata --

    def _load_stats(self) -> dict:
        stats_path = self.data_root / "aggregated_stats.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                return json.load(f)
        return {}

    def _stat_values(self, agg_key: str) -> DatasetStatisticalValues:
        """Build DatasetStatisticalValues from aggregated_stats.json entry."""
        entry = self._stats.get(agg_key, {})
        mean = np.array(entry.get("mean", [0.0]))
        std = np.array(entry.get("std", [1.0]))
        mn = np.array(entry.get("min", mean - 3 * std))
        mx = np.array(entry.get("max", mean + 3 * std))
        return DatasetStatisticalValues(
            mean=mean,
            std=std,
            min=mn,
            max=mx,
            q01=mn,
            q99=mx,
        )

    def _build_metadata(self) -> DatasetMetadata:
        state_stats: dict[str, DatasetStatisticalValues] = {}
        state_modalities: dict[str, StateActionMetadata] = {}
        for s in self.selected_states:
            top, sub = s.split(".")
            key_name = f"{top}_{sub}"
            agg_key = f"obs/agent/{top}/{sub}"
            sv = self._stat_values(agg_key)
            state_stats[key_name] = sv
            state_modalities[key_name] = StateActionMetadata(
                absolute=True,
                rotation_type=None,
                shape=(len(sv.mean),),
                continuous=True,
            )

        action_stats: dict[str, DatasetStatisticalValues] = {}
        action_modalities: dict[str, StateActionMetadata] = {}
        rel_keys_set = {
            k.replace(".", "_") for k in self.relative_action_keys
        }
        for a in self.selected_actions:
            top, sub = a.split(".")
            key_name = f"{top}_{sub}"
            # Relative mode: use joint_pos_rel stats for relative keys
            if self.relative_action and key_name in rel_keys_set:
                agg_key = f"actions/joint_pos_rel/{sub}"
                if agg_key not in self._stats:
                    agg_key = f"actions/{top}/{sub}"
            else:
                agg_key = f"actions/{top}/{sub}"
            sv = self._stat_values(agg_key)
            action_stats[key_name] = sv
            action_modalities[key_name] = StateActionMetadata(
                absolute=True,
                rotation_type=None,
                shape=(len(sv.mean),),
                continuous=True,
            )

        video_modalities: dict[str, VideoMetadata] = {}
        for cam in self.selected_observations:
            video_modalities[cam] = VideoMetadata(
                resolution=self._native_video_resolution,
                channels=3,
                fps=float(self.fps),
            )

        return DatasetMetadata(
            statistics=DatasetStatistics(state=state_stats, action=action_stats),
            modalities=DatasetModalities(
                video=video_modalities,
                state=state_modalities,
                action=action_modalities,
            ),
            embodiment_tag=self.tag,
        )

    # -- Sharding --

    def _generate_shards(self) -> tuple[list[list[int]], np.ndarray]:
        total_steps = int(self._trajectory_lengths.sum())
        n_shards = max(1, int(np.ceil(total_steps / self.num_steps_per_shard)))
        cutoffs = np.linspace(0, total_steps, n_shards + 1)[1:]
        shards: list[list[int]] = [[]]
        shard_lengths: list[int] = []
        cum = 0
        curr_shard = 0
        last_cum = 0
        for tid in range(len(self.traj_lengths)):
            shards[-1].append(tid)
            cum += self.traj_lengths[tid]
            if curr_shard < len(cutoffs) and cum > cutoffs[curr_shard]:
                shards.append([])
                curr_shard += 1
                shard_lengths.append(cum - last_cum)
                last_cum = cum
        shard_lengths.append(cum - last_cum)
        if len(shards[-1]) == 0:
            shards.pop()
        while len(shard_lengths) > len(shards):
            shard_lengths.pop()
        while len(shard_lengths) < len(shards):
            shard_lengths.append(0)
        print(f"MjThor: generated {len(shards)} shards")
        return shards, np.array(shard_lengths)

    # -- Shard caching --

    def start_cache_shard(self, shard_index: int) -> None:
        self._cache_job = self._executor.submit(
            self._cache_shard_impl,
            shard_index,
        )

    def finish_cache_shard(self):
        assert self._cache_job is not None
        result = self._cache_job.result()
        self.cached_video_frames = result[0]
        self.cached_states = result[1]
        self.cached_actions = result[2]
        self.cached_languages = result[3]
        self.shard_start_indices = result[4]
        self._cache_job = None

    def delete_cached_shard(self):
        self.cached_video_frames = None
        self.cached_states = None
        self.cached_actions = None
        self.cached_languages = None
        self.shard_start_indices = None

    def get_trajectories_in_shard(self) -> list[int]:
        assert self.shard_start_indices is not None
        return list(self.shard_start_indices.keys())

    def _cache_shard_impl(self, shard_index: int):
        """Cache all data for a shard."""
        print(f"MjThor: caching shard {shard_index}")
        t0 = time.time()
        traj_ids = self.sharded_trajectories[shard_index]

        cached_video: dict[int, dict[str, np.ndarray]] = {}
        cached_st: dict[int, dict[str, np.ndarray]] = {}
        cached_act: dict[int, dict[str, np.ndarray]] = {}
        cached_lang: dict[int, str] = {}
        start_indices: dict[int, int] = {}
        cum = 0

        for tid in traj_ids:
            start_indices[tid] = cum
            traj_len = self.traj_lengths[tid]
            file_traj_idx = self.traj_file_traj_idxs[tid]
            h5_path = self.data_root / self.traj_files[tid]
            max_frame = traj_len + self._max_delta_index + 1

            # Video -- skip entire trajectory if any camera fails
            vid_ok = True
            traj_video: dict[str, np.ndarray] = {}
            for cam in self.selected_observations:
                try:
                    with h5py.File(h5_path, "r") as f:
                        obs_ds = f[f"traj_{file_traj_idx}/obs/sensor_data/{cam}"]
                        vname = obs_ds[:].tobytes().decode("utf-8").rstrip("\x00")
                    vpath = str(h5_path.with_name(vname))
                    vr = VideoReader(vpath)
                    nf = min(len(vr), max_frame)
                    if nf == 0:
                        raise ValueError("empty video")
                    batch = vr.get_batch(list(range(nf)))
                    if hasattr(batch, "asnumpy"):
                        frames = batch.asnumpy()
                    elif hasattr(batch, "numpy"):
                        frames = batch.numpy()
                    else:
                        frames = np.asarray(batch)
                    if frames.ndim != 4:
                        raise ValueError(f"unexpected frame shape {frames.shape}")
                    traj_video[cam] = frames
                except Exception as e:
                    print(f"WARN: skipping traj {tid} – video failed {cam}: {e}")
                    vid_ok = False
                    break

            if not vid_ok:
                cum += traj_len
                continue

            cached_video[tid] = traj_video

            # State, Action, Language from HDF5
            try:
                with h5py.File(h5_path, "r") as f:
                    obs_scene = json.loads(
                        f[f"traj_{file_traj_idx}/obs_scene"][()]
                        .decode("utf-8")
                        .rstrip("\x00")
                    )
                    cached_lang[tid] = obs_scene.get("task_description", "")

                    # State
                    agent_grp = f[f"traj_{file_traj_idx}/obs/agent"]
                    state_top_keys = set(
                        s.split(".")[0] for s in self.selected_states
                    )
                    state_decoded: dict[str, list[dict]] = {
                        k: [] for k in state_top_keys
                    }
                    for tk in state_top_keys:
                        ds = agent_grp[tk]
                        n = min(ds.shape[0], max_frame)
                        for i in range(n):
                            raw = ds[i].tobytes().decode("utf-8").rstrip("\x00")
                            state_decoded[tk].append(json.loads(raw))

                    cached_st[tid] = {}
                    for s in self.selected_states:
                        top, sub = s.split(".")
                        kn = f"{top}_{sub}"
                        vals: list[list[float]] = []
                        for d in state_decoded[top]:
                            v = d.get(sub)
                            if v is None:
                                vals.append(vals[-1] if vals else [0.0])
                            elif isinstance(v, list):
                                vals.append(v)
                            else:
                                vals.append([float(v)])
                        arr = np.array(vals, dtype=np.float32)
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 1)
                        cached_st[tid][kn] = arr

                    # Action
                    act_grp = f[f"traj_{file_traj_idx}/actions"]
                    act_top_keys = set(
                        a.split(".")[0] for a in self.selected_actions
                    )
                    # Relative mode always uses source joint_pos_rel (never convert in-code)
                    if self.relative_action:
                        act_top_keys.add("joint_pos_rel")
                    act_decoded: dict[str, list[dict]] = {
                        k: [] for k in act_top_keys
                    }
                    for tk in act_top_keys:
                        if tk not in act_grp:
                            continue
                        ds = act_grp[tk]
                        n = min(ds.shape[0], max_frame + 1)
                        for i in range(n):
                            raw = ds[i].tobytes().decode("utf-8").rstrip("\x00")
                            act_decoded[tk].append(json.loads(raw))

                    cached_act[tid] = {}
                    rel_keys_set = {
                        k.replace(".", "_") for k in self.relative_action_keys
                    }
                    for a in self.selected_actions:
                        top, sub = a.split(".")
                        kn = f"{top}_{sub}"
                        # Relative mode: always use source joint_pos_rel for relative keys
                        load_top = top
                        if (
                            self.relative_action
                            and kn in rel_keys_set
                            and "joint_pos_rel" in act_decoded
                            and len(act_decoded["joint_pos_rel"]) > 0
                        ):
                            load_top = "joint_pos_rel"
                        vals: list[list[float]] = []
                        source_list = act_decoded.get(load_top) or act_decoded[top]
                        for d in source_list:
                            v = d.get(sub)
                            if v is None:
                                vals.append(vals[-1] if vals else [0.0])
                            elif isinstance(v, list):
                                vals.append(v)
                            else:
                                vals.append([float(v)])
                        arr = np.array(vals, dtype=np.float32)
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 1)
                        cached_act[tid][kn] = arr

            except Exception as e:
                print(f"WARN: skipping traj {tid} – HDF5 read failed: {e}")
                cached_video.pop(tid, None)
                cached_st.pop(tid, None)
                cached_act.pop(tid, None)
                cached_lang.pop(tid, None)
                cum += traj_len
                continue

            cum += traj_len

        elapsed = time.time() - t0
        print(f"MjThor: cached shard {shard_index} in {elapsed:.1f}s")
        return cached_video, cached_st, cached_act, cached_lang, start_indices

    # -- Multi-chunk expansion (mirrors DROID ShardedLeRobotSubLangSingleActionChunkDatasetDROID) --

    def _expand_video_chunks(
        self, anchor: int, traj_len: int
    ) -> np.ndarray | None:
        """Expand anchor into ``8*num_chunks + 1`` video frame indices.

        Within each 24-step chunk we subsample 8 frames at stride 3
        (offsets ``[0, 3, 6, 9, 12, 15, 18, 21]``), then append one
        trailing frame after the last chunk.  This matches the DROID
        ``_uniform_sample_from_language_ranges`` for video.
        """
        max_frames = 8 * self.max_chunk_size + 1
        offsets = [0, 3, 6, 9, 12, 15, 18, 21]
        sampled: list[int] = []

        def _add(a: int) -> None:
            if a < 0 or a + 23 >= traj_len:
                return
            if len(sampled) + len(offsets) > max_frames:
                return
            for o in offsets:
                sampled.append(a + o)

        _add(anchor)
        step = 1
        bk_done = fwd_done = False
        while len(sampled) < max_frames and (not bk_done or not fwd_done):
            if not bk_done:
                ba = anchor - 24 * step
                if ba < 0:
                    bk_done = True
                else:
                    _add(ba)
            if len(sampled) >= max_frames:
                break
            if not fwd_done:
                fa = anchor + 24 * step
                if fa >= traj_len:
                    fwd_done = True
                else:
                    _add(fa)
            step += 1

        if not sampled:
            return None
        arr = np.array(sorted(set(sampled)), dtype=int)
        if arr.size > max_frames:
            arr = arr[:max_frames]

        last = arr[-1]
        trailing = last + 3
        if trailing < traj_len and arr.size < max_frames:
            arr = np.append(arr, trailing)
        else:
            if arr.size <= 8:
                return None
            arr = arr[:-7]

        if arr.size % 8 != 1:
            return None
        return arr

    def _expand_action_chunks(
        self, anchor: int, traj_len: int, num_chunks: int
    ) -> np.ndarray | None:
        """Expand anchor into ``24 * num_chunks`` action indices."""
        max_frames = 24 * num_chunks
        sampled: list[int] = []

        def _add(a: int) -> None:
            if a < 0 or a + 24 >= traj_len:
                return
            if len(sampled) + 24 > max_frames:
                return
            if num_chunks > 0 and len(sampled) // 24 >= num_chunks:
                return
            for o in range(24):
                sampled.append(a + o)

        _add(anchor)
        step = 1
        bk_done = fwd_done = False
        while len(sampled) < max_frames and (not bk_done or not fwd_done):
            if num_chunks > 0 and len(sampled) // 24 >= num_chunks:
                break
            if not bk_done:
                ba = anchor - 24 * step
                if ba < 0:
                    bk_done = True
                else:
                    _add(ba)
            if len(sampled) >= max_frames:
                break
            if not fwd_done:
                fa = anchor + 24 * step
                if fa >= traj_len:
                    fwd_done = True
                else:
                    _add(fa)
            step += 1

        if not sampled:
            return None
        arr = np.array(sorted(set(sampled)), dtype=int)
        remainder = arr.size % 24
        if remainder:
            arr = arr[:arr.size - remainder]
        if arr.size == 0:
            return None
        return arr

    def _expand_state_chunks(
        self, anchor: int, traj_len: int, num_chunks: int
    ) -> np.ndarray | None:
        """Expand anchor into ``num_chunks`` state anchor indices (1 per chunk)."""
        sampled: list[int] = []

        def _add(a: int) -> None:
            if a < 0 or a + 24 >= traj_len:
                return
            if len(sampled) >= num_chunks:
                return
            sampled.append(a)

        _add(anchor)
        step = 1
        bk_done = fwd_done = False
        while len(sampled) < num_chunks and (not bk_done or not fwd_done):
            if not bk_done:
                ba = anchor - 24 * step
                if ba < 0:
                    bk_done = True
                else:
                    _add(ba)
            if len(sampled) >= num_chunks:
                break
            if not fwd_done:
                fa = anchor + 24 * step
                if fa >= traj_len:
                    fwd_done = True
                else:
                    _add(fa)
            step += 1

        if not sampled:
            return None
        return np.array(sorted(set(sampled)), dtype=int)

    def _retrieve_and_pad(
        self, arr: np.ndarray, indices: np.ndarray, padding: str = "zero"
    ) -> np.ndarray:
        """Index into ``arr`` with OOB handling (mirrors DROID retrieve_data_and_pad)."""
        n = arr.shape[0]
        safe = np.clip(indices, 0, max(n - 1, 0))
        out = arr[safe].copy()
        if padding == "first_last":
            out[indices < 0] = arr[0]
            out[indices >= n] = arr[-1]
        else:
            out[indices < 0] = 0
            out[indices >= n] = 0
        return out

    # -- get_step_data --

    def get_step_data(
        self,
        trajectory_id: int,
        indices: dict[str, np.ndarray],
    ) -> dict[str, Any] | None:
        """Return raw (un-transformed) data dict for one step.

        When ``max_chunk_size > 0`` the sample is expanded to multiple
        action-chunks (matching the DROID multi-chunk behaviour).
        """
        assert self.cached_video_frames is not None, "Shard not cached"

        if trajectory_id not in self.cached_video_frames:
            return None

        data: dict[str, Any] = {}
        traj_len = self.traj_lengths[trajectory_id]

        # ---- Multi-chunk path ----
        if self.max_chunk_size > 0:
            first_vid_key = next(
                (k for k in indices if k.startswith("video.")), None
            )
            if first_vid_key is None:
                return None
            anchor = int(indices[first_vid_key][0])

            vid_idx = self._expand_video_chunks(anchor, traj_len)
            if vid_idx is None or vid_idx.size == 0:
                return None
            num_chunks = (vid_idx.size - 1) // 8

            act_idx = self._expand_action_chunks(anchor, traj_len, num_chunks)
            if act_idx is None or act_idx.size == 0:
                return None
            st_idx = self._expand_state_chunks(anchor, traj_len, num_chunks)
            if st_idx is None or st_idx.size == 0:
                return None

            # Video
            for cam in self.selected_observations:
                key = f"video.{cam}"
                frames = self.cached_video_frames[trajectory_id][cam]
                safe = np.clip(vid_idx, 0, max(frames.shape[0] - 1, 0))
                data[key] = frames[safe]

            # State
            for s in self.selected_states:
                top, sub = s.split(".")
                kn = f"{top}_{sub}"
                key = f"state.{kn}"
                arr = self.cached_states[trajectory_id][kn]
                data[key] = self._retrieve_and_pad(arr, st_idx, "first_last")

            # Action
            for a in self.selected_actions:
                top, sub = a.split(".")
                kn = f"{top}_{sub}"
                key = f"action.{kn}"
                arr = self.cached_actions[trajectory_id][kn]
                act_data = self._retrieve_and_pad(arr, act_idx, "zero")
                # Relative mode uses source joint_pos_rel only; no in-code conversion
                data[key] = act_data

            # Language
            lang_key = "annotation.language.task_description"
            if lang_key in indices:
                data[lang_key] = self.cached_languages[trajectory_id]

            for v in data.values():
                if hasattr(v, "__len__") and len(v) == 0:
                    return None
            return data

        # ---- Single-step path (original behaviour, max_chunk_size == 0) ----
        for cam in self.selected_observations:
            key = f"video.{cam}"
            if key not in indices:
                continue
            all_frames = self.cached_video_frames[trajectory_id][cam]
            n = all_frames.shape[0] if all_frames.ndim >= 1 else 1
            step_idx = np.clip(indices[key], 0, max(n - 1, 0))
            data[key] = all_frames[step_idx]

        for s in self.selected_states:
            top, sub = s.split(".")
            kn = f"{top}_{sub}"
            key = f"state.{kn}"
            if key not in indices:
                continue
            arr = self.cached_states[trajectory_id][kn]
            n = arr.shape[0] if arr.ndim >= 1 else 1
            step_idx = np.clip(indices[key], 0, max(n - 1, 0))
            data[key] = arr[step_idx]

        for a in self.selected_actions:
            top, sub = a.split(".")
            kn = f"{top}_{sub}"
            key = f"action.{kn}"
            if key not in indices:
                continue
            arr = self.cached_actions[trajectory_id][kn]
            n = arr.shape[0] if arr.ndim >= 1 else 1
            step_idx = np.clip(indices[key], 0, max(n - 1, 0))
            data[key] = arr[step_idx]

        lang_key = "annotation.language.task_description"
        if lang_key in indices:
            data[lang_key] = self.cached_languages[trajectory_id]

        for v in data.values():
            if hasattr(v, "__len__") and len(v) == 0:
                return None
        return data


# ---------------------------------------------------------------------------
# Mixture dataset
# ---------------------------------------------------------------------------


class MjThorMixtureDataset(IterableDataset):
    """Iterable mixture dataset for MjThor data, compatible with the
    DreamZero training loop.

    Provides the ``merged_metadata`` attribute required by
    BaseExperiment.setup and works with get_train_dataloader.
    """

    def __init__(
        self,
        datasets: list[MjThorSingleDataset],
        weights: list[float] | None = None,
        training: bool = True,
        seed: int = 42,
        shard_sampling_rate: float = 0.5,
        num_shards_to_sample: int = 2**20,
    ):
        super().__init__()
        self.datasets = datasets
        self.training = training
        self.seed = seed
        self.shard_sampling_rate = shard_sampling_rate
        self.num_shards_to_sample = num_shards_to_sample

        if weights is None:
            total = sum(len(d) for d in datasets)
            weights = [len(d) / total for d in datasets]
        w = np.array(weights, dtype=np.float64)
        w /= w.sum()
        self._dataset_weights = w

        # Build shard schedule
        self._all_shards: list[tuple[int, int]] = []
        shard_w: list[float] = []
        for di, ds in enumerate(self.datasets):
            dw = self._dataset_weights[di]
            normed = ds.shard_lengths / max(ds.shard_lengths.sum(), 1e-12)
            for si in range(ds.num_shards):
                self._all_shards.append((di, si))
                shard_w.append(float(normed[si] * dw))
        sw = np.array(shard_w, dtype=np.float64)
        sw /= sw.sum()
        self._shard_weights = sw
        self._schedule = self._generate_schedule()

        # Distributed info
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.worker_id: int | None = None
        self.num_workers: int | None = None

        self._build_merged_metadata()

        for ds in self.datasets:
            if training:
                ds.transforms.train()
            else:
                ds.transforms.eval()

    # -- Factory classmethod for Hydra --

    @classmethod
    def from_config(
        cls,
        data_roots: list[str],
        selected_states: list[str],
        selected_actions: list[str],
        selected_observations: list[str],
        embodiment_tag: str = "mjthor",
        modality_configs: dict[str, Any] | None = None,
        transforms: ComposedModalityTransform | None = None,
        house_idxs: list[int] | None = None,
        drop_n_last_frames: int = 0,
        num_steps_per_shard: int = 10_000,
        img_size: list[int] | None = None,
        training: bool = True,
        seed: int = 42,
        shard_sampling_rate: float = 0.5,
        num_shards_to_sample: int = 2**20,
        max_chunk_size: int = 0,
        relative_action: bool = False,
        relative_action_keys: list[str] | None = None,
    ) -> "MjThorMixtureDataset":
        resolved_img = tuple(img_size) if img_size else None
        datasets = []
        for root in data_roots:
            ds = MjThorSingleDataset(
                data_root=root,
                selected_states=selected_states,
                selected_actions=selected_actions,
                selected_observations=selected_observations,
                embodiment_tag=embodiment_tag,
                modality_configs=modality_configs or {},
                transforms=transforms,
                house_idxs=house_idxs,
                drop_n_last_frames=drop_n_last_frames,
                num_steps_per_shard=num_steps_per_shard,
                img_size=resolved_img,
                max_chunk_size=max_chunk_size,
                relative_action=relative_action,
                relative_action_keys=relative_action_keys or [],
            )
            datasets.append(ds)
        return cls(
            datasets=datasets,
            training=training,
            seed=seed,
            shard_sampling_rate=shard_sampling_rate,
            num_shards_to_sample=num_shards_to_sample,
        )

    # -- Schedule and distribution --

    def _generate_schedule(self) -> list[tuple[int, int]]:
        rng = np.random.default_rng(self.seed)
        ids = rng.choice(
            len(self._all_shards),
            size=self.num_shards_to_sample,
            p=self._shard_weights,
        )
        schedule = [self._all_shards[i] for i in ids]
        if self.training:
            rng.shuffle(schedule)
        return schedule

    def reset_seed(self, seed: int):
        self.seed = seed
        self._schedule = self._generate_schedule()

    def _filter_schedule(self) -> list[tuple[int, int]]:
        worker_info = get_worker_info()
        wid = worker_info.id if worker_info else 0
        nw = worker_info.num_workers if worker_info else 1
        if self.worker_id is None:
            self.worker_id = wid
            self.num_workers = nw
        return [
            s
            for i, s in enumerate(self._schedule)
            if i % (self.world_size * nw) == self.rank * nw + wid
        ]

    # -- Metadata --

    def _build_merged_metadata(self):
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        for ds in self.datasets:
            tag_val = ds.tag.value
            if tag_val not in self.merged_metadata:
                self.merged_metadata[tag_val] = ds.metadata

    # -- Iteration --

    def __iter__(self):
        schedule = self._filter_schedule()
        rng = np.random.default_rng(self.seed)

        for si, (di, shard_idx) in enumerate(schedule):
            ds = self.datasets[di]
            ds.start_cache_shard(shard_idx)
            ds.finish_cache_shard()

            if si + 1 < len(schedule):
                next_di, next_si = schedule[si + 1]
                self.datasets[next_di].start_cache_shard(next_si)

            all_steps: list[tuple[int, int]] = []
            for tid in ds.get_trajectories_in_shard():
                traj_idx = ds.get_trajectory_index(tid)
                allowed = ds.step_filter[tid]
                max_step = ds.trajectory_lengths[traj_idx] - ds.max_delta_index
                allowed = allowed[allowed < max_step]
                for step in allowed:
                    all_steps.append((tid, int(step)))

            if self.training:
                rng.shuffle(all_steps)
            n_sample = int(ds.num_steps_per_shard * self.shard_sampling_rate)
            for tid, step in all_steps[:n_sample]:
                step_indices = {
                    key: di_arr + step
                    for key, di_arr in ds.delta_indices.items()
                }
                step_data = ds.get_step_data(tid, step_indices)
                if step_data is not None:
                    yield ds.transforms(step_data)

            ds.delete_cached_shard()

    def __len__(self) -> int:
        total = 0
        for di, _ in self._schedule:
            ds = self.datasets[di]
            total += int(ds.num_steps_per_shard * self.shard_sampling_rate)
        return total

    def __str__(self) -> str:
        lines = ["MjThorMixtureDataset:"]
        for i, ds in enumerate(self.datasets):
            lines.append(
                f"  [{i}] {ds.data_root} - "
                f"{len(ds)} steps, {ds.num_shards} shards"
            )
        return "\n".join(lines)
