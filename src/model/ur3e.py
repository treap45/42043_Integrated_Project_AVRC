from pathlib import Path

import mujoco
import numpy as np

JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
N_JOINTS = len(JOINT_NAMES)

DEFAULT_XML = Path(__file__).resolve().parents[2] / "assets/controller_model.xml"


class UR3eModel:
    """Kinematics and dynamics for the independent controller-side model."""

    def __init__(self, xml_path: Path = DEFAULT_XML):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # model.joint(name) raises KeyError if a joint is missing, so no separate check is needed.
        joint_ids = [self.model.joint(name).id for name in JOINT_NAMES]
        self.qpos_adr = self.model.jnt_qposadr[joint_ids].copy()
        self.dof_adr = self.model.jnt_dofadr[joint_ids].copy()

    @staticmethod
    def _as_joint_vector(value, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (N_JOINTS,):
            raise ValueError(f"{name} must have shape ({N_JOINTS},), got {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        return vector

    def _set_state(self, q, qd=None) -> None:
        q = self._as_joint_vector(q, "q")
        if qd is None:
            qd = np.zeros(N_JOINTS)
        else:
            qd = self._as_joint_vector(qd, "qd")

        self.data.qpos[self.qpos_adr] = q
        self.data.qvel[self.dof_adr] = qd
        mujoco.mj_forward(self.model, self.data)

    def _set_configuration(self, q) -> None:
        """Position-level computation only: enough for frame poses and Jacobians."""
        self.data.qpos[self.qpos_adr] = self._as_joint_vector(q, "q")
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)  # camera poses are not updated by mj_kinematics

    def mass_matrix(self, q) -> np.ndarray:
        """Return the 6x6 joint-space inertia matrix M(q), including armature."""
        self._set_state(q)

        full_matrix = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, full_matrix)
        return full_matrix[np.ix_(self.dof_adr, self.dof_adr)].copy()

    def bias_forces(self, q, qd=None) -> np.ndarray:
        """Return MuJoCo qfrc_bias: gravity plus velocity-dependent terms."""
        self._set_state(q, qd)
        return self.data.qfrc_bias[self.dof_adr].copy()

    def gravity_torque(self, q) -> np.ndarray:
        """Return the torque needed to balance gravity at q."""
        return self.bias_forces(q, np.zeros(N_JOINTS))

    def forward_dynamics(self, q, qd, tau) -> np.ndarray:
        """Predict qdd for the controller model, ignoring contacts and saturation."""
        tau = self._as_joint_vector(tau, "tau")
        M = self.mass_matrix(q)
        h = self.bias_forces(q, qd)
        return np.linalg.solve(M, tau - h)

    def site_pose(self, q, site_name: str = "tcp") -> tuple[np.ndarray, np.ndarray]:
        """Return a named site's world-frame position and rotation matrix."""
        self._set_configuration(q)
        position = self.data.site(site_name).xpos.copy()
        rotation = self.data.site(site_name).xmat.reshape(3, 3).copy()
        return position, rotation

    def camera_pose(self, q, camera_name: str = "wrist_cam") -> tuple[np.ndarray, np.ndarray]:
        """Return a camera's world-frame position and rotation matrix.

        MuJoCo camera convention: the camera looks along its local -z axis with +y up.
        """
        self._set_configuration(q)
        camera = self.model.camera(camera_name).id
        position = self.data.cam_xpos[camera].copy()
        rotation = self.data.cam_xmat[camera].reshape(3, 3).copy()
        return position, rotation

    def site_jacobian(self, q, site_name: str = "tcp") -> np.ndarray:
        """Return the 6x6 geometric Jacobian of a site: rows [linear; angular] in the world frame."""
        self._set_configuration(q)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.model.site(site_name).id)
        return np.vstack([jacp[:, self.dof_adr], jacr[:, self.dof_adr]])