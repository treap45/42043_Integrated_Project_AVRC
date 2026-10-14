"""Controller-model check: assets/controller_model.xml against the simulated robot in assets/scene.xml.

Kinematics must match exactly (same frames for every q). Dynamics differ only by the unmodelled
end-effector payload; this script reports how large that mismatch is, joint by joint.

Test script: evaluates both models at the same q directly, outside the pipeline.

Run from the repo root with the venv active:
    python scripts/check_controller_model.py
"""

from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PLANT_XML = REPO_ROOT / "assets/scene.xml"
CTRL_XML = REPO_ROOT / "assets/controller_model.xml"
N_ARM = 6
N_SAMPLES = 500
JOINTS = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]


def evaluate(model, data, q):
    """Frames, gravity torque and arm inertia matrix at q (zero velocity)."""
    data.qpos[:] = model.qpos0
    data.qpos[:N_ARM] = q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    frames = {}
    for site in ("attachment_site", "tcp"):
        frames[site] = (data.site(site).xpos.copy(), data.site(site).xmat.reshape(3, 3).copy())
    cam = model.camera("wrist_cam").id
    frames["wrist_cam"] = (data.cam_xpos[cam].copy(), data.cam_xmat[cam].reshape(3, 3).copy())
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    return frames, data.qfrc_bias[:N_ARM].copy(), M[:N_ARM, :N_ARM].copy()


def rot_err(R1, R2):
    # Frobenius norm of the difference; arccos of the trace angle loses precision near zero.
    return np.linalg.norm(R1 - R2)


def main():
    plant = mujoco.MjModel.from_xml_path(str(PLANT_XML))
    ctrl = mujoco.MjModel.from_xml_path(str(CTRL_XML))
    pd, cd = mujoco.MjData(plant), mujoco.MjData(ctrl)

    print("== 1. Model ==")
    arm_mass = lambda m: sum(m.body_mass[m.body(n).id] for n in   # noqa: E731
                             ["shoulder_link", "upper_arm_link", "forearm_link",
                              "wrist_1_link", "wrist_2_link", "wrist_3_link"])
    payload = sum(plant.body_subtreemass[plant.body(n).id] for n in ["flange"])
    print(f"  controller model: nq={ctrl.nq} nv={ctrl.nv}, armature {ctrl.dof_armature}")
    print(f"  moving arm links: plant {arm_mass(plant):.3f} kg, controller {arm_mass(ctrl):.3f} kg")
    print(f"  unmodelled payload (everything below the flange in the plant): {payload:.3f} kg")

    rng = np.random.default_rng(0)
    qs = rng.uniform(-np.pi, np.pi, size=(N_SAMPLES, N_ARM))
    qs[0] = plant.key("observe").qpos[:N_ARM]

    pos_errs, rot_errs, dg, dM_rel = [], [], [], []
    for q in qs:
        fp, gp, Mp = evaluate(plant, pd, q)
        fc, gc, Mc = evaluate(ctrl, cd, q)
        pos_errs.append(max(np.linalg.norm(fp[k][0] - fc[k][0]) for k in fp))
        rot_errs.append(max(rot_err(fp[k][1], fc[k][1]) for k in fp))
        dg.append(gp - gc)
        dM_rel.append((np.diag(Mp) - np.diag(Mc)) / np.diag(Mp))
    dg, dM_rel = np.array(dg), np.array(dM_rel)

    print(f"\n== 2. Kinematics ({N_SAMPLES} random q; flange, tcp, wrist_cam) ==")
    kin_ok = max(pos_errs) < 1e-9 and max(rot_errs) < 1e-9
    print(f"  max position error {max(pos_errs):.2e} m, max rotation matrix error {max(rot_errs):.2e}  "
          f"{'PASS' if kin_ok else 'FAIL'}")

    print("\n== 3. Payload mismatch (plant minus controller model) ==")
    print(f"  {'joint':8s} {'gravity at observe':>20s} {'max |gravity| diff':>20s} {'diag M diff (median)':>22s}")
    for j, name in enumerate(JOINTS):
        print(f"  {name:8s} {dg[0, j]:17.3f} Nm {np.abs(dg[:, j]).max():17.3f} Nm "
              f"{100 * np.median(dM_rel[:, j]):20.1f} %")

    print("\n== Summary ==")
    print(f"  kinematics   {'PASS' if kin_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
