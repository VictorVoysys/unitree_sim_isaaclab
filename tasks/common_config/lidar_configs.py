# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Lidar sensor configuration for the G1 (and H1-2) heads.

The G1 ships with a Livox MID-360 attached to the torso. Two facts about
the physical mount drive the offset baked into these presets:

  1. URDF mid360_joint says xyz=(0.0002835, 0.00003, 0.41618) on
     torso_link, rpy=(0, 0.04014, 0).
  2. The unit on the real robot is mounted UPSIDE-DOWN — the LIVOX
     branding faces the floor. That's a 180° roll about X relative to
     what the URDF describes. Multiple community projects flag this
     (e.g. deepglint/FAST_LIO_LOCALIZATION_HUMANOID), and we've
     verified geometrically against arm-teleop poses in
     plugins/unitree_g1/visualize_lidar_fov.py — without the roll, the
     lidar cone points at the ceiling and sees no arm work at all.

So the sensor's local +Z (its dome direction) points DOWN in torso
frame after applying the offset rotation here, matching the physical
robot. The xr_teleoperate URDF in github.com/unitreerobotics is
missing this roll; if the sim's USD asset is regenerated from that
URDF the offset here will need re-checking.

The MID-360's vertical FOV is asymmetric: -7° (below dome direction)
to +52° (above dome direction). After the dome-down flip in robot
frame, the cone covers ~+7° above robot horizon to ~-52° below — the
range that matters for chest/table/face teleop work.
"""

import math

from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
from isaaclab.utils import configclass

# MultiMeshRayCaster routes prim_expr through Python's re module, so
# negative lookahead works. We exclude the lidar's own mesh
# (mid360_link) from the cast list because in the G1 USD that mesh
# extends ~0.41 m from the URDF link origin (the optical center),
# so rays cast outward from the optical center hit the sensor's own
# housing first and never reach the world. Excluding mid360_link
# lets the rays escape and pick up real environment + arm hits.
# Tighter regex: match URDF link prims only (anything ending in
# "_link", plus the bare "pelvis" root link), and explicitly exclude
# the two sensor links whose mesh either doesn't exist (d435_link) or
# self-collides with the rays (mid360_link). Excluding the non-link
# children (Looks, joints, root_joint, PhysicsMaterial, imu_in_*)
# matters because track_mesh_transforms=True requires every matched
# prim to be Xformable, and those scopes/materials aren't.
# NOTE: track_mesh_transforms is INTENTIONALLY False here, even though
# robot links move every step. With track=True the multi-mesh
# raycaster refreshes ~50 link transforms each step, and on this
# hardware that visibly blocks the env's main control loop — the sim
# stops stepping (locomotion never starts, FPS check times out, no
# new lidar/camera SHM updates). track=False makes the cloud see the
# robot frozen at its initial pose, but the table/walls/floor are
# still seen correctly, which is what we need for now.
#
# Restoring live arm tracking is future work. Options:
#   - profile track=True to find the bottleneck and optimise; or
#   - drop down to a per-arm MultiMeshRayCaster bound to a single
#     "left_arm" / "right_arm" articulation with track=True and a
#     much smaller mesh count.
# Exclude the central body block (torso + head + sensor housings + pelvis
# + waist links) because those are the meshes directly under the dome —
# rays exit the lidar mount and hit them at ~0.41 m before reaching the
# scene. We keep the limb links (shoulders, elbows, wrists, hands,
# hips, knees, ankles, plus the pelvis_contour visual) so arms still
# show up in the cloud.
_ROBOT_MESHES_EXCLUDING_LIDAR = MultiMeshRayCasterCfg.RaycastTargetCfg(
    prim_expr=(
        "/World/envs/env_.*/Robot/"
        "(?!"
        "mid360_link$|"
        "d435_link$|"
        "torso_link$|"
        "head_link$|"
        "pelvis$|"
        "waist_(yaw|roll|pitch)_link$"
        ").+_link"
    ),
    track_mesh_transforms=False,
)


# Quaternion (w, x, y, z) for R = Ry(0.04014) · Rx(π), the URDF
# convention (Rz·Ry·Rx with rpy = (π, 0.04014, 0)). The 180° roll
# about X dominates and is what the module docstring's "dome-down"
# discussion is about; the small Y pitch is the URDF's mid360_joint
# forward tilt. Bridge and visualize_lidar_fov.py in the plugin repo
# both use this same convention — see sim_lidar_bridge_tcp.py
# (LIDAR_RPY_TORSO = (π, 0.04014, 0), _rot_xyz = Rz·Ry·Rx).
_LIDAR_OFFSET_ROT_WXYZ = (0.0, 0.99979, 0.0, -0.02007)
_LIDAR_OFFSET_POS = (0.0002835, 0.00003, 0.41618)


# Lidar ROI presets in sensor-local frame (after the dome-down offset
# rotation). Each entry is ((vertical_fov_min, vertical_fov_max),
# (horizontal_fov_min, horizontal_fov_max)) in degrees.
#
#   "camera": cropped to the D435 head-camera frustum. HFOV 69.4°
#     → azimuth ±34.7°. VFOV 42.5° centred at camera pitch 47.6°
#     → elevation [26.35°, 68.85°], clipped to [26.35°, 52°] by the
#     MID-360's hardware vertical FOV (the bottom ~17° of camera
#     vertical extent is outside the lidar cone — nothing to simulate
#     there).
#   "full": the entire MID-360 hardware FOV (-7° to +52° elevation,
#     360° azimuth). Use for downstream consumers that need wide
#     coverage (SLAM, navigation); pay for it with a higher
#     points_per_scan than the cropped default.
_ROI_FOV_DEG = {
    "camera": ((26.35, 52.0), (-34.7, 34.7)),
    "full":   ((-7.0, 52.0), (-180.0, 180.0)),
}


@configclass
class LidarPresets:
    """Lidar preset configurations for Unitree humanoid heads."""

    @classmethod
    def g1_mid360(
        cls,
        prim_path: str = "/World/envs/env_.*/Robot/torso_link",
        update_period: float = 0.10,
        max_distance: float = 30.0,
        roi: str = "camera",
        points_per_scan: int = 1680,
        mesh_prim_paths: list = None,
    ) -> MultiMeshRayCasterCfg:
        """Approximate a MID-360 attached to the G1 torso, dome-down.

        Uses MultiMeshRayCaster so we can hit dynamic meshes (the
        robot's own arms, the manipulated object) in addition to the
        static scene. Pattern is repetitive (Velodyne-style channels +
        azimuth grid) — a simplification of the real non-repetitive
        rosette — but per-frame point density is tuned to match the
        real MID-360 within the chosen ROI.

        Args:
            roi: Region of the lidar dome to simulate. ``"camera"``
                (default) crops to the D435 head-camera frustum
                (~7% of the dome) so we don't waste rays on directions
                the operator's depth panel can't see. ``"full"``
                simulates the entire MID-360 FOV; use for downstream
                consumers that need wide coverage (SLAM, navigation).
                See ``_ROI_FOV_DEG`` for the exact ranges.
            points_per_scan: Target number of rays per scan. The
                default ``1680`` is real-MID-360-in-frustum throughput
                at the ``"camera"`` ROI (manufacturer spec: 200 kpts/s
                × ~7.3 % solid-angle ratio × 100 ms ≈ 1.46 k, with a
                small bump for typical rosette hotspots). For
                ``"full"`` ROI, ``20000`` matches real-hardware
                full-dome rate; bump there only if you actually need
                full-dome density. Channels and horizontal angular
                resolution are derived from this value and the chosen
                ROI's aspect ratio so angular spacing stays roughly
                equal in both directions; actual rendered ray count
                rounds to ±2 % of the requested target.
        """
        try:
            vfov, hfov = _ROI_FOV_DEG[roi]
        except KeyError:
            raise ValueError(
                f"roi must be one of {sorted(_ROI_FOV_DEG)}; got {roi!r}"
            )
        if points_per_scan < 1:
            raise ValueError(
                f"points_per_scan must be >= 1; got {points_per_scan}"
            )

        # Derive channels and horizontal_res from the requested point
        # budget, keeping vertical and horizontal angular spacing
        # roughly equal: total ≈ channels × h_count, h_count ≈
        # aspect × channels, so channels ≈ sqrt(total / aspect).
        vspan = vfov[1] - vfov[0]
        hspan = hfov[1] - hfov[0]
        aspect = hspan / vspan
        channels = max(1, round(math.sqrt(points_per_scan / aspect)))
        h_count = max(1, round(points_per_scan / channels))
        horizontal_res = hspan / h_count

        if mesh_prim_paths is None:
            # Only prims that exist in EVERY G1 task we care about. The
            # MultiMeshRayCaster validates these paths at scene-build
            # time and crashes the whole sim if any are missing — so
            # task-specific objects (cylinders, blocks, extra tables)
            # must be opted in by the task's env_cfg via the
            # mesh_prim_paths argument, not added here.
            #
            # Robot meshes are listed via a negative-lookahead regex
            # that excludes mid360_link (the lidar's own housing), so
            # the rays don't self-collide on the dome before reaching
            # the world. See _ROBOT_MESHES_EXCLUDING_LIDAR above.
            mesh_prim_paths = [
                _ROBOT_MESHES_EXCLUDING_LIDAR,
                "/World/envs/env_.*/PackingTable",
                "/World/envs/env_.*/Room",
            ]
        return MultiMeshRayCasterCfg(
            prim_path=prim_path,
            update_period=update_period,
            offset=MultiMeshRayCasterCfg.OffsetCfg(
                pos=_LIDAR_OFFSET_POS,
                rot=_LIDAR_OFFSET_ROT_WXYZ,
            ),
            attach_yaw_only=False,
            ray_alignment="base",
            max_distance=max_distance,
            mesh_prim_paths=mesh_prim_paths,
            pattern_cfg=LidarPatternCfg(
                channels=channels,
                vertical_fov_range=vfov,
                horizontal_fov_range=hfov,
                horizontal_res=horizontal_res,
            ),
            debug_vis=False,
        )
