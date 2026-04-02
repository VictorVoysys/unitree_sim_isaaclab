# Testing Oden (unitree_g1) with the Isaac Lab Simulator

This document describes how to connect the Oden teleoperation stack (`/home/victor/unitree_g1`) to the Isaac Lab simulation environment (`/home/victor/unitree_sim_isaaclab`) for virtual testing of the Unitree G1 robot.

## Overview

Both systems communicate via CycloneDDS. The simulator exposes the same DDS topics as the physical G1 robot, so Oden Streamer's G1 plugin can talk to it as if it were a real robot.

```
Simulator (Isaac Lab)              Oden Streamer (G1 plugin)
─────────────────────              ────────────────────────
publishes rt/lowstate        ───►  subscribes rt/lowstate (IMU, joints)
consumes  rt/lowcmd          ◄───  sends commands via LocoClient RPC
consumes  rt/run_command     ◄───  sends velocity/yaw locomotion commands
WebRTC / virtual video       ───►  captures from /dev/video0
```

## Prerequisites

- Isaac Sim 4.5.0 or 5.0/5.1 + Isaac Lab installed
- Oden builds successfully (`cargo build` in unitree_g1)
- Python dependencies for the simulator (`pip install -r requirements.txt`)

## Step 1: Fix the DDS domain mismatch

The simulator uses DDS domain **1**, but the Oden G1 bridge uses domain **0**. They must match or DDS messages won't be exchanged.

**File to change:** `/home/victor/unitree_g1/plugins/unitree_g1/unitree_g1_sys/cpp/g1_bridge.cpp` line 54

```cpp
// Change from:
unitree::robot::ChannelFactory::Instance()->Init(0, network_interface);
// To:
unitree::robot::ChannelFactory::Instance()->Init(1, network_interface);
```

Then rebuild Oden:

```bash
cd /home/victor/unitree_g1
cargo build
```

## Step 2: Start the simulator

Use a **wholebody task** so that LocoClient velocity commands (from Oden's gamepad teleop) are supported.

```bash
cd /home/victor/unitree_sim_isaaclab
python sim_main.py \
  --device cpu \
  --enable_cameras \
  --task Isaac-MoveCylinder-G129-Dex1-Wholebody \
  --enable_dex1_dds \
  --robot_type g129
```

The first startup may take a while as Isaac Sim loads resources.

Verify in the console output that DDS is initialized on channel 1.

## Step 3: Set up video streaming

Oden Streamer expects to capture video from `/dev/video0` (configured in `g1_streamer.vproj`). Choose one of these options:

### Option A: Virtual video device (recommended for full integration)

Install v4l2loopback to create a virtual camera, then pipe simulator frames into it:

```bash
sudo apt install v4l2loopback-dkms
sudo modprobe v4l2loopback devices=1 video_nr=0
```

Then configure the simulator to output frames to this device (may require custom scripting with the teleimager module or ffmpeg).

### Option B: Separate video streams (simpler)

Run the simulator in headless mode with WebRTC streaming and view video separately:

```bash
python sim_main.py \
  --no_render \
  --livestream_type 2 \
  --public_ip 127.0.0.1 \
  --task Isaac-MoveCylinder-G129-Dex1-Wholebody \
  --enable_dex1_dds \
  --robot_type g129
```

View the WebRTC stream in a browser, and use Oden only for control (not video).

## Step 4: Start Oden Streamer

```bash
cd /home/victor/unitree_g1
cargo run --bin oden_streamer -- --project plugins/unitree_g1/g1_streamer.vproj
```

Key plugin parameters (configured in `g1_streamer.vproj` or at runtime):
- `g1_network_interface`: set to `lo` (loopback) if both run on the same machine
- `g1_max_vx`: max forward velocity (default 0.8 m/s)
- `g1_max_vy`: max lateral velocity (default 0.4 m/s)
- `g1_max_vyaw`: max yaw rate (default 0.8 rad/s)

## Step 5: Start OdenVR (operator side)

```bash
cd /home/victor/unitree_g1
cargo run --bin oden_ui
```

Connect to the Streamer instance. Use a gamepad to send locomotion commands.

## Verification checklist

- [ ] DDS domain matches (both using domain 1)
- [ ] Simulator is running and printing state updates
- [ ] Oden Streamer starts without DDS errors
- [ ] `rt/lowstate` messages are received by Oden (check G1 telemetry in Oden GUI)
- [ ] Gamepad commands from OdenVR reach the simulator
- [ ] Video stream is available (via WebRTC or virtual camera)

## Troubleshooting

### No DDS communication
- Confirm both processes use the same DDS domain ID (1)
- If on the same machine, set `g1_network_interface` to `lo`
- Check that CycloneDDS is installed and `CYCLONEDDS_URI` is not overriding the domain

### LocoClient commands not recognized
- Use a wholebody task (e.g. `Isaac-MoveCylinder-*-Wholebody`) — non-wholebody tasks only accept joint-level `rt/lowcmd`
- Verify the simulator implements the LocoClient RPC service endpoint

### Simulator runs slowly
- Use `--device cuda:0` instead of `--device cpu` if a GPU is available
- Reduce camera load with `--camera_exclude` or disable cameras
- Lower physics rate with `--step_hz 50`
