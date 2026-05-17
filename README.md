# Robotic Arm Manipulation Learning on SO-ARM101

Vision-guided manipulation and imitation learning on the Hiwonder SO-ARM101, built on NVIDIA Jetson AGX Orin. The project combines low-level hardware control, dual-camera perception, dataset collection, and learning-based policy training for tasks such as pick, place, stack, and workspace organization.

## Overview

This repository is organized around two complementary workflows:

1. Real robot control and system bring-up
2. Data collection and model training for visuomotor policies

The current system supports:

- Feetech STS/SCS bus control for the full SO-ARM101 arm
- Gripper control on the same servo bus
- Dual-camera capture for wrist and stand viewpoints
- Scripted pick-and-place execution using vision
- Dataset collection for imitation learning
- Training pipelines for Diffusion Policy and ACT
- Leader-follower teleoperation using two physical SO-ARM101 arms

## Current Platform

- Compute: NVIDIA Jetson AGX Orin
- Robot: Hiwonder SO-ARM101
- Servos: Feetech STS/SCS, IDs 1-6
- Cameras: wrist-mounted USB camera and stand-mounted USB camera
- Python: 3.10

### Dual-Arm Teleoperation Setup

The current teleoperation workflow uses two complete arms:

- Leader arm on `/dev/ttyACM0`
  Leader torques are disabled so the arm can be moved by hand.
- Follower arm on `/dev/ttyACM1`
  Follower torques remain enabled and mirror the leader pose.

Verified joint mapping for both arms:

- ID1: base rotation
- ID2: shoulder
- ID3: elbow
- ID4: wrist pitch
- ID5: wrist roll
- ID6: gripper

Important operating note:

- ID5 wrist roll has the largest usable rotation range.
- Shoulder and base motion are mechanically constrained by the arm geometry and table contact, so physical leader motion may not always span the follower's full reachable workspace.

## What Works Today

- Low-level serial communication with Feetech servos through `hardware/feetech_bus.py`
- High-level arm control, FK, and basic IK through `hardware/arm_controller.py`
- Gripper open, close, width control, and simple grasp detection
- Dual-camera capture and camera configuration management
- Orange detection pipeline for object localization experiments
- Scripted pick execution for system validation before model deployment
- Keyboard-based dataset collection through `teleop.py`
- Policy training through `train.py`
- Policy and scripted inference through `inference.py`
- Leader-follower teleoperation through `scripts/teleop.py`

## Repository Layout

```text
roboarm/
├── config/
│   ├── arm_config.yaml
│   └── camera_config.yaml
├── data/
│   └── dataset.py
├── hardware/
│   ├── arm_controller.py
│   ├── camera_manager.py
│   ├── feetech_bus.py
│   ├── gripper_controller.py
│   └── ik_solver.py
├── perception/
│   └── orange_detector.py
├── policy/
│   ├── act_policy.py
│   ├── diffusion_policy.py
│   └── networks.py
├── scripts/
│   ├── teleop.py
│   ├── test_hardware.py
│   ├── view_cameras.py
│   ├── visualise_dataset.py
│   └── utility and diagnostic scripts
├── calibrate.py
├── inference.py
├── teleop.py
└── train.py
```

### Entry Points

- `scripts/teleop.py`
  Dual-arm leader-follower teleoperation. Reads live servo positions from the leader and mirrors them to the follower.
- `teleop.py`
  Keyboard-driven episode collection pipeline for imitation learning datasets.
- `train.py`
  Training entry point for Diffusion Policy and ACT.
- `inference.py`
  Scripted execution or learned-policy rollout on the robot.
- `calibrate.py`
  Camera intrinsics and hand-eye calibration utilities.

## Setup

Install dependencies:

```bash
cd ~/Documents/roboarm
pip install -r requirements.txt
```

Serial permissions are currently handled manually after reboot:

```bash
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
```

If you are using the Jetson environment described in this project, the Python environment is expected to provide access to Torch, OpenCV, and NumPy.

## Quick Start

### 1. Smoke-test hardware

```bash
cd ~/Documents/roboarm
python scripts/test_hardware.py
```

### 2. Check cameras

```bash
cd ~/Documents/roboarm
python scripts/view_cameras.py
```

### 3. Run dual-arm teleoperation

```bash
cd ~/Documents/roboarm
python scripts/teleop.py
```

Useful options:

```bash
python scripts/teleop.py --rate_hz 12 --deadband 0 --speed 50
python scripts/teleop.py --leader /dev/ttyACM0 --follower /dev/ttyACM1
```

### 4. Collect demonstrations for learning

```bash
cd ~/Documents/roboarm
python teleop.py --episodes 20 --save_dir datasets
```

### 5. Visualize dataset quality

```bash
cd ~/Documents/roboarm
python scripts/visualise_dataset.py --data_dir datasets
```

### 6. Train a policy

```bash
cd ~/Documents/roboarm
python train.py --policy diffusion --data_dir datasets --epochs 300 --batch_size 64
python train.py --policy act --data_dir datasets --epochs 200 --batch_size 32
```

### 7. Run scripted or learned inference

```bash
cd ~/Documents/roboarm
python inference.py --mode scripted
python inference.py --mode policy --policy diffusion --ckpt checkpoints/diffusion_best.pth
python inference.py --mode policy --policy act --ckpt checkpoints/act_best.pth
```

## Learning Pipeline

The long-term objective is robust visuomotor learning for manipulation tasks such as:

- picking isolated objects
- stacking objects
- sorting and organizing tabletop scenes
- learning reusable grasp and place primitives from demonstrations

Current training support includes:

- Diffusion Policy for chunked action prediction from visual and proprioceptive observations
- ACT for transformer-based action chunking from demonstrations
- episode-based dataset loading from compressed `.npz` recordings
- dual-camera observations plus proprioceptive state as policy input

## Current Limitations

- Serial port permissions are not yet persistent across reboot
- Mechanical workspace is limited by table contact and arm geometry
- Leader motion cannot always express the follower's entire reachable envelope
- Camera calibration values are still configuration-driven and should be refined with measured calibration sessions
- The repository currently includes both production paths and exploratory diagnostic scripts; this is intentional during active bring-up

## Near-Term Roadmap

- persist device permissions with udev rules
- improve teleoperation smoothness and follower tracking fidelity
- collect a larger demonstration dataset with consistent task labels
- train and benchmark policies on pick, stack, and organize tasks
- improve evaluation tooling for rollout success rate and failure categorization

## Reference Documentation

Detailed implementation history and milestone notes are available in `docs/PROJECT_HISTORY.md`.
