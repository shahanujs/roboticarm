# Phase 1: Teleoperation Data Collection Pipeline (ROS2 + HDF5)

This module implements a production-grade ROS2 data collection node for imitation learning.
It synchronizes camera images, joint states, and expert action vectors into a structured HDF5 dataset at 20-50 Hz.

## Architecture Design

### Data Flow

1. ROS2 subscribers ingest:
   - `sensor_msgs/msg/Image` (camera frames)
   - `sensor_msgs/msg/JointState` (joint positions/velocities)
   - `std_msgs/msg/Float64MultiArray` (expert action vector)
2. `ApproximateTimeSynchronizer` aligns the three streams by timestamp.
3. Callback rate limiter enforces configurable logging frequency (`target_hz`).
4. Validator checks joint/action dimensions and image format.
5. HDF5 writer appends:
   - Raw modalities: image, joint position, joint velocity, action, synchronized timestamps
   - Normalized vectors: z-score normalized joint/action arrays
   - Metadata and normalization statistics
6. File is flushed periodically and on graceful shutdown.

### Timing and Synchronization Strategy

- Synchronization: message_filters ApproximateTime policy
- Queue size: configurable (default 200)
- Allowed inter-stream skew (`slop_sec`): configurable (default 0.025 sec)
- Logging rate gate: enforced by collector callback using ROS time deltas

### Dataset Layout

```
/<root>
  attrs:
    schema_version = "1.0"
    created_utc = "..."
    image_topic, joint_state_topic, action_topic
    target_hz, slop_sec, frame_id

  /observations
    images                uint8   [N, H, W, C]
    joint_positions       float32 [N, J]
    joint_velocities      float32 [N, J]
    joint_names           S       [J]

  /actions
    expert                float32 [N, A]

  /normalized
    joint_positions_z     float32 [N, J]
    joint_velocities_z    float32 [N, J]
    expert_actions_z      float32 [N, A]

  /timestamps
    image                 float64 [N]
    joint_state           float64 [N]
    action                float64 [N]
    synced                float64 [N]

  /stats
    joints_pos_mean       float64 [J]
    joints_pos_std        float64 [J]
    joints_vel_mean       float64 [J]
    joints_vel_std        float64 [J]
    actions_mean          float64 [A]
    actions_std           float64 [A]
```

## Directory Structure

```
anujroboticsarm/
  phase1_teleop_data/
    README.md
    requirements.txt
    config/
      collector_config.yaml
    scripts/
      run_collector.py
    teleop_hdf5_collector/
      __init__.py
      config.py
      normalization.py
      hdf5_writer.py
      collector_node.py
```

## Run Instructions

1. Source ROS2 environment:
   - `source /opt/ros/humble/setup.bash` or `source /opt/ros/jazzy/setup.bash`
2. Install Python deps:
   - `pip install -r requirements.txt`
3. Start collector:
   - `python scripts/run_collector.py --config config/collector_config.yaml`

## Notes for LeRobot Compatibility

- The HDF5 schema is intentionally explicit and easy to map to LeRobot dataset conversion scripts.
- If needed, add a post-processing converter to emit LeRobot-native keys while preserving raw archives.
