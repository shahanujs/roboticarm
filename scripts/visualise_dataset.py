"""
Visualise dataset statistics and replay an episode.
Usage: python scripts/visualise_dataset.py --data_dir datasets/
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def visualise(data_dir: str) -> None:
    paths = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    print(f"Found {len(paths)} episodes")

    all_joints = []
    for p in paths:
        ep = np.load(p)
        all_joints.append(ep["act_joints"])

    all_joints = np.concatenate(all_joints, axis=0)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    joint_names = ["base","shoulder","elbow","wrist_pitch","wrist_roll","wrist_yaw"]
    for i, (ax, name) in enumerate(zip(axes.flat, joint_names)):
        ax.hist(all_joints[:, i], bins=40)
        ax.set_title(name)
        ax.set_xlabel("deg")
    plt.tight_layout()
    out = os.path.join(data_dir, "joint_histograms.png")
    plt.savefig(out, dpi=100)
    print(f"Saved histograms → {out}")

    # Replay first episode
    ep = np.load(paths[0])
    T  = ep["wrist_imgs"].shape[0]
    print(f"Replaying episode {paths[0]}  ({T} steps)")
    for t in range(T):
        w = cv2.resize(ep["wrist_imgs"][t], (320, 240))
        s = cv2.resize(ep["stand_imgs"][t], (320, 240))
        disp = np.hstack([w, s])
        cv2.putText(disp, f"step {t+1}/{T}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
        cv2.imshow("Episode Replay", disp)
        if cv2.waitKey(60) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="datasets")
    args = ap.parse_args()
    visualise(args.data_dir)
