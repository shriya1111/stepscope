"""Where is the arm in space? Joint angles -> 3D positions, using a MuJoCo model.

Policies for arms like the SO-101 or ALOHA output joint targets. Safety rules
are about space ("stay out of this zone"), so the gate and the safety scoring
need the 3D position of the arm that a set of joint angles would produce. This
is forward kinematics; it involves no physics and takes a few microseconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Kinematics:
    """Maps joint values to the 3D positions of chosen bodies on the robot.

    model: a mujoco.MjModel (or a path to an MJCF/URDF file).
    joints: names of the joints that your joint vector refers to, in order.
    bodies: names of the bodies to track (e.g. each link and the gripper).
    links: optional pairs of indices into `bodies` to draw as arm segments;
        defaults to a chain through `bodies` in order.
    """

    def __init__(self, model, joints: list[str], bodies: list[str], links: list[list[int]] | None = None, base_qpos=None):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model)) if isinstance(model, (str, Path)) else model
        self.data = mujoco.MjData(self.model)
        self.joints, self.bodies = list(joints), list(bodies)
        self._qadr = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints])
        jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
        self._bid = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies])
        if (self._bid < 0).any() or min(jid) < 0:
            raise ValueError("unknown joint or body name")
        self.joint_low = self.model.jnt_range[jid, 0].copy()
        self.joint_high = self.model.jnt_range[jid, 1].copy()
        self.links = links if links is not None else [[i, i + 1] for i in range(len(bodies) - 1)]
        self._base = np.array(self.model.qpos0 if base_qpos is None else base_qpos, dtype=float)

    def points(self, q) -> np.ndarray:
        """(n_bodies, 3) positions for one joint vector."""
        self.data.qpos[:] = self._base
        self.data.qpos[self._qadr] = np.asarray(q, dtype=float)
        self._mj.mj_kinematics(self.model, self.data)
        return self.data.xpos[self._bid].copy()

    def path(self, qs) -> np.ndarray:
        """(T, n_bodies, 3) positions for a sequence of joint vectors."""
        return np.stack([self.points(q) for q in np.asarray(qs, dtype=float)])


def project_to_camera(points: np.ndarray, cam_pos, cam_mat, fovy_deg: float, width: int, height: int) -> np.ndarray:
    """Pixel coordinates (..., 2) of 3D points (..., 3) in a MuJoCo camera's image.

    MuJoCo cameras look along their own -z axis, with x to the right and y up.
    """
    rel = (np.asarray(points, dtype=float) - np.asarray(cam_pos, dtype=float)) @ np.asarray(cam_mat, dtype=float).reshape(3, 3)
    f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
    depth = np.maximum(-rel[..., 2], 1e-6)
    return np.stack([width / 2 + f * rel[..., 0] / depth, height / 2 - f * rel[..., 1] / depth], axis=-1)
