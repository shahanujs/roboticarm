# Project History and System Notes

## Project Goal

Build a complete robotic manipulation stack on the Hiwonder SO-ARM101 that can progress from direct teleoperation to data collection and finally to learned visuomotor policies for pick, place, stack, and tabletop organization tasks.

The engineering goal is not only to run a robot arm, but to build a system that is understandable end to end:

- hardware communication
- joint control
- camera perception
- calibration
- data collection
- policy training
- real-robot inference

## Hardware Platform

- Compute: NVIDIA Jetson AGX Orin
- Robot hardware: two Hiwonder SO-ARM101 arms during teleoperation development
- Servo bus: Feetech STS/SCS half-duplex serial bus at 1,000,000 baud
- Servo IDs: 1 through 6 on each arm
- Cameras: two USB cameras, one wrist-mounted and one stand-mounted

## System Architecture

### 1. Low-level bus control

The first layer of the project is `hardware/feetech_bus.py`, which implements the Feetech STS/SCS packet protocol directly.

Completed work:

- packet send and receive
- checksum handling
- read and write operations
- servo ping
- goal position writes
- sync write for multi-servo updates
- present-position reads in signed and unsigned form
- torque enable and disable support

Why this matters:

- it establishes direct hardware control without depending on a vendor GUI
- it enables deterministic multi-joint updates
- it exposes the exact servo state required for teleoperation and logging

### 2. High-level arm control

The next layer is `hardware/arm_controller.py`.

Completed work:

- YAML-driven joint configuration
- tick-to-degree and degree-to-tick conversion
- joint-space command APIs
- sync-write based joint updates
- home and observe poses
- approximate forward kinematics
- basic IK pathway through `ikpy`
- workspace-aware motion routines used by higher-level scripts

Why this matters:

- it turns raw servo control into robot-centric joint control
- it provides a stable interface for scripted motion and learning-based control

### 3. Gripper control

The gripper is handled separately in `hardware/gripper_controller.py`.

Completed work:

- open and close control
- width control by fractional opening
- grasp detection heuristic using servo state and motion completion
- support for shared or dedicated bus access

Why this matters:

- gripper state becomes part of both demonstrations and learned action outputs

### 4. Camera pipeline

The camera stack provides wrist and stand views for operator feedback, scripted picking, and dataset collection.

Completed work:

- dual-camera configuration in `config/camera_config.yaml`
- camera access through `hardware/camera_manager.py`
- live visualization through `scripts/view_cameras.py`
- camera intrinsics and hand-eye calibration tooling in `calibrate.py`

Why this matters:

- both teleoperation recording and learned policies depend on synchronized visual context

### 5. Perception

The initial perception path is object detection focused, currently oriented around orange detection.

Completed work:

- HSV-based orange detection
- support for pose estimation using camera calibration
- scripted pipeline integration through `inference.py`

Why this matters:

- it provides a practical bridge between pure motion bring-up and autonomous task execution

### 6. Learning pipeline

The ML stack is designed around imitation learning.

Completed work:

- keyboard-based episode collection in `teleop.py`
- `.npz` episode format for observations and actions
- dataset loader in `data/dataset.py`
- Diffusion Policy training path in `train.py`
- ACT training path in `train.py`
- policy inference runner in `inference.py`

Why this matters:

- it creates a full path from demonstration to deployable policy rollout

## Teleoperation Development History

The project later expanded from single-arm motion testing into a proper leader-follower setup using two separate SO-ARM101 arms.

### Initial clarification

An important system clarification was made during bring-up:

- the gripper servo is not a separate leader device
- the platform uses two complete physical arms
- one arm acts as the human-operated leader
- the second arm acts as the motor-driven follower

This changed the teleoperation design completely and aligned the project with the standard leader-follower imitation learning workflow.

### Leader and follower roles

- Leader arm: `/dev/ttyACM0`
  Torques disabled, positions read continuously while the user moves the arm by hand.
- Follower arm: `/dev/ttyACM1`
  Torques enabled, positions commanded to match the leader.

### Verified servo identification

Leader joint identification was completed empirically by disabling torque and moving one physical joint at a time.

Final verified mapping:

- ID1 = base rotation
- ID2 = shoulder
- ID3 = elbow
- ID4 = wrist pitch
- ID5 = wrist roll
- ID6 = gripper

Important observations from testing:

- ID5 provides the largest practical rotation range
- none of the arm joints besides wrist roll should be treated as fully continuous in normal operation
- shoulder and elbow motion can induce secondary movement in neighboring joints because the arm is under gravity when torque is off
- physical table contact limits how far the leader can be folded in some poses

### Teleoperation implementation

The current leader-follower implementation is `scripts/teleop.py`.

Completed work:

- reads all six unsigned servo positions from leader
- disables torque on leader on startup
- enables torque on follower on startup
- captures a stable startup pose from the live leader state
- sets the follower startup pose from that captured leader pose
- mirrors all six joints using sync-write updates
- supports exact mirroring by default with `deadband=0`
- disables follower torques on shutdown for safe exit

Operational note:

- teleoperation is close to direct mirroring but not mathematically perfect because the mechanism itself has friction, backlash, speed limits, and workspace constraints

## Configuration and Current Known Values

### Arm configuration

Current arm configuration is stored in `config/arm_config.yaml`.

Key values:

- arm bus default: `/dev/ttyACM0`
- gripper ID: `6`
- gripper open position: `2400`
- gripper close position: `1600`
- active Cartesian control DOF: `4`

### Camera configuration

Current camera configuration is stored in `config/camera_config.yaml`.

Key values:

- wrist camera device ID: `2`
- stand camera device ID: `0`
- observation resize: `96 x 96`

## Problems Solved During Bring-Up

### Serial permissions

Problem:

- `/dev/ttyACM0` and `/dev/ttyACM1` required manual permission changes after reboot

Current workaround:

```bash
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
```

Planned improvement:

- replace manual permission updates with persistent udev rules

### Joint identification ambiguity

Problem:

- initial uncertainty about which physical joint corresponded to which servo ID

Resolution:

- each joint was moved independently with live position monitoring until the full mapping was confirmed

### Mechanical safety around the wrist camera

Problem:

- large wrist roll motion can bring the camera body close to the arm structure depending on arm pose

Current handling:

- teleoperation and testing are done with attention to arm posture and conservative movement speed

## Current State of the Project

The project now has a complete vertical slice from hardware access to learning code.

Already implemented:

- direct servo bus communication
- joint-level and gripper-level control
- dual-camera support
- calibration tools
- perception experiments
- scripted pick execution
- dataset collection format
- learning pipelines for Diffusion Policy and ACT
- dual-arm leader-follower teleoperation

Not finished yet:

- persistent device permissions
- large-scale demonstration collection
- benchmarked policy training runs on meaningful manipulation tasks
- robust multi-object stacking and organization behaviors

## Recommended Next Milestones

1. Collect high-quality teleoperated demonstrations with task labels and consistent reset conditions.
2. Add persistent udev rules so the system survives reboot cleanly.
3. Benchmark teleoperation repeatability and follower tracking error.
4. Train baseline Diffusion Policy and ACT models on a narrow manipulation task first.
5. Expand from pick-and-place to stacking and multi-object organization.

## Technical Points to Be Ready to Explain

- Why sync-write is preferred over sequential single-servo writes for teleoperation
- Why servo positions are handled in ticks at the hardware layer and degrees at the robot-control layer
- Why leader torque must be disabled for natural manual demonstration
- Why follower torque must remain enabled during mirroring
- Why startup pose capture matters for gripper alignment and initial synchronization
- Why dual-camera observations are useful for manipulation learning
- Why imitation learning needs consistent demonstration quality and task coverage
- Why a scripted baseline is valuable before learning-based deployment