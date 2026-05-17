"""
Compatibility wrapper.

Use strict input-follow implementation by default.
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(script_dir, "gripper_input_arm_follow.py")
    print("gripper_guided_arm.py is deprecated; forwarding to gripper_input_arm_follow.py")
    cmd = [sys.executable, target] + sys.argv[1:]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
