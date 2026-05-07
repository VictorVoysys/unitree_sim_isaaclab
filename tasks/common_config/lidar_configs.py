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


# Quaternion (w, x, y, z) for R = Rx(π) · Ry(0.04014). See module
# docstring for why; computed by hand and re-verified by:
#   plugins/unitree_g1/visualize_lidar_fov.py with --no-roll-fix off.
# The 180° roll about X dominates; the small Y pitch is the URDF's
# mid360_joint forward tilt.
_LIDAR_OFFSET_ROT_WXYZ = (0.0, 0.99979, 0.0, 0.02007)
_LIDAR_OFFSET_POS = (0.0002835, 0.00003, 0.41618)


@configclass
class LidarPresets:
    """Lidar preset configurations for Unitree humanoid heads."""

    @classmethod
    def g1_mid360(
        cls,
        prim_path: str = "/World/envs/env_.*/Robot/torso_link",
        # MVP density: light enough to not steal more than ~1 fps from
        # the 15 fps sim baseline. 16 channels × 1° azimuth × 10 Hz
        # = 5760 rays/scan × 10 = 57.6 kray/s. Bumped back up later if
        # we need finer detail; for verifying the geometry is right
        # the current setting is more than enough.
        update_period: float = 0.10,
        max_distance: float = 30.0,
        channels: int = 16,
        horizontal_res: float = 1.0,
        mesh_prim_paths: list = None,
    ) -> MultiMeshRayCasterCfg:
        """Approximate a MID-360 attached to the G1 torso, dome-down.

        Uses MultiMeshRayCaster so we can hit dynamic meshes (the robot's
        own arms, the manipulated object) in addition to the static
        scene. The pattern is repetitive (Velodyne-style channels +
        azimuth grid) which is a simplification of the real
        non-repetitive rosette — point density per scan is in the same
        ballpark for VR-teleop visualisation.
        """
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
                vertical_fov_range=(-7.0, 52.0),
                horizontal_fov_range=(-180.0, 180.0),
                horizontal_res=horizontal_res,
            ),
            debug_vis=False,
        )
