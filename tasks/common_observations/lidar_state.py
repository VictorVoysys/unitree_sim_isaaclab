# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Lidar observation: pulls the simulated MID-360 point cloud from the
scene each step and publishes it to a shared-memory region for the
streamer/visualiser to consume.

Mirrors the structure of camera_state.py (single global writer, async
queue, throttled to a publish rate), but writes a [N, 3] float32 cloud
in the lidar's LOCAL frame rather than RGB images. The consumer
applies the lidar-in-torso extrinsic if it wants robot-frame output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import os
import sys
import threading
import queue

import torch

# Allow `from tools...` import the same way camera_state.py does.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tools.shared_memory_utils import PointCloudWriter

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_writer = PointCloudWriter()
_state = {
    "frame_step": 0,
    "publish_every_steps": 2,   # at 100 Hz step rate this is 50 Hz publish
    "sensor_key": None,         # auto-detected first time
    "async_started": False,
    "queue": None,
    "thread": None,
    "call_count": 0,
}


def _async_writer_loop(q: "queue.Queue", writer: PointCloudWriter):
    while True:
        item = q.get()
        if item is None:
            break
        try:
            pts, frame_id = item
            writer.write(pts, frame_id=frame_id)
        except Exception as e:
            print(f"[lidar_state] async write error: {e}")


def _ensure_async_started():
    if _state["async_started"]:
        return
    _state["queue"] = queue.Queue(maxsize=1)
    _state["thread"] = threading.Thread(
        target=_async_writer_loop,
        args=(_state["queue"], _writer),
        daemon=True,
    )
    _state["thread"].start()
    _state["async_started"] = True


def _find_lidar_sensor(env):
    """Pick whichever sensor in the scene exposes a ray-caster cloud.

    Looks across both `env.scene[name]` (entity name lookup, which works
    for any entity bucket — articulations, sensors, extras) and
    `env.scene.sensors`. Caches after first hit.

    Returns the sensor object on success, None until both lookups fail
    AND we've logged at least one warning so misconfigurations show up
    rather than silently returning empty clouds.
    """
    cached = _state.get("sensor_key")
    if cached is not None:
        try:
            return env.scene[cached]
        except Exception:
            pass

    # Prefer the named entity lookup first — that's what the scene-
    # config `mid360_lidar = LidarPresets.g1_mid360()` entry produces.
    candidate_names = ["mid360_lidar", "lidar", "mid360"]
    for name in candidate_names:
        try:
            sensor = env.scene[name]
        except (KeyError, AttributeError):
            continue
        if sensor is None:
            continue
        data = getattr(sensor, "data", None)
        if data is None:
            # Sensor exists but hasn't been updated yet. Cache the key
            # so we don't keep looking it up; future calls will see
            # data populated.
            _state["sensor_key"] = name
            return sensor
        _state["sensor_key"] = name
        return sensor

    # Fallback: scan env.scene.sensors for anything ray-caster-like.
    sensors = getattr(env.scene, "sensors", None)
    if sensors:
        for name in sensors.keys():
            sensor = sensors[name]
            data = getattr(sensor, "data", None)
            if data is not None and getattr(data, "ray_hits_w", None) is not None:
                _state["sensor_key"] = name
                return sensor

    if not _state.get("warned_no_sensor"):
        print("[lidar_state] WARNING: could not find lidar sensor in env.scene; "
              "lidar SHM will not be published. Checked names: "
              f"{candidate_names}", flush=True)
        _state["warned_no_sensor"] = True
    return None


def get_lidar_cloud(env: "ManagerBasedRLEnv"):
    """ObsTerm-compatible wrapper. Returns a placeholder tensor (the
    observation manager wants ONE per term) and pushes the cloud to SHM
    as a side effect.
    """
    _state["frame_step"] = (_state["frame_step"] + 1) % max(1, _state["publish_every_steps"])
    if _state["frame_step"] != 0:
        return torch.zeros((1, 1), device=env.device)

    sensor = _find_lidar_sensor(env)
    if sensor is None:
        return torch.zeros((1, 1), device=env.device)

    data = sensor.data
    if data is None:
        if not _state.get("warned_no_data"):
            print("[lidar_state] WARNING: sensor.data is None on call",
                  _state.get("call_count", 0), flush=True)
            _state["warned_no_data"] = True
        return torch.zeros((1, 1), device=env.device)

    if getattr(data, "ray_hits_w", None) is None:
        if not _state.get("warned_no_hits"):
            print("[lidar_state] WARNING: data.ray_hits_w is None on call",
                  _state.get("call_count", 0), flush=True)
            _state["warned_no_hits"] = True
        return torch.zeros((1, 1), device=env.device)

    if not _state.get("logged_first"):
        print(f"[lidar_state] first valid call: sensor='{_state['sensor_key']}', "
              f"ray_hits_w shape={tuple(data.ray_hits_w.shape)}", flush=True)
        _state["logged_first"] = True

    # Critical: data.pos_w / data.quat_w are the PARENT PRIM's pose (the
    # one cfg.prim_path resolves to), NOT the lidar's optical-center
    # pose. cfg.offset.{pos,rot} aren't baked into data.{pos,quat}_w.
    # Compose the actual lidar pose here so the cloud we publish is in
    # true lidar-local frame (optical center at origin, dome direction
    # at +Z, after the URDF offset and the dome-down roll).
    parent_pos_w = data.pos_w[0]
    parent_quat_w = data.quat_w[0]
    cfg_offset_pos = torch.tensor(list(sensor.cfg.offset.pos),
                                  device=parent_pos_w.device, dtype=parent_pos_w.dtype)
    cfg_offset_rot = torch.tensor(list(sensor.cfg.offset.rot),
                                  device=parent_pos_w.device, dtype=parent_pos_w.dtype)
    pos_w = parent_pos_w + quat_apply(parent_quat_w, cfg_offset_pos)
    quat_w = quat_mul(parent_quat_w, cfg_offset_rot)

    hits_w = data.ray_hits_w[0]            # [B, 3]

    # Filter out misses. Isaac Lab returns the sensor origin (or pos_w +
    # max_distance ray) for rays that didn't hit — both look like a
    # uniform shell. Easier filter: drop points farther than the
    # sensor's max_distance minus a small epsilon.
    rel = hits_w - pos_w.unsqueeze(0)
    dist = torch.linalg.norm(rel, dim=1)
    max_dist = float(getattr(sensor.cfg, "max_distance", 30.0))
    mask = (dist > 0.05) & (dist < max_dist - 1e-3)
    hits_w_kept = hits_w[mask]

    if hits_w_kept.shape[0] == 0:
        return torch.zeros((1, 1), device=env.device)

    # Transform world hits → lidar local frame. quat_apply_inverse takes
    # (q_wxyz, v_world) and rotates v into the frame defined by q.
    rel = hits_w_kept - pos_w.unsqueeze(0)
    quat_batch = quat_w.unsqueeze(0).expand(rel.shape[0], -1)
    pts_local = quat_apply_inverse(quat_batch, rel)

    pts_np = pts_local.detach().cpu().numpy().astype("float32", copy=False)

    _ensure_async_started()
    q = _state["queue"]
    try:
        if q.full():
            q.get_nowait()
        q.put_nowait((pts_np, "lidar"))
    except Exception:
        pass

    return torch.zeros((1, 1), device=env.device)
