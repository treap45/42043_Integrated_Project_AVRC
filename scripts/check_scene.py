"""Scene check: loads assets/scene.xml, renders the wrist camera and an overview, and runs a short
physics sanity test.

This is a test script, not part of the pipeline: it reads simulator ground truth freely and holds the
arm with MuJoCo's own bias forces, which the controller must not do.

Run from the repo root with the venv active:
    python scripts/check_scene.py
"""

from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = REPO_ROOT / "assets/scene.xml"
OUT_DIR = REPO_ROOT / "results/check_scene"
TABLE_Z = 0.75
CUBE_HALF = 0.025


def summary(model):
    print("== 1. Model ==")
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} nmocap={model.nmocap} ncam={model.ncam}")
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        print(f"  actuator {model.actuator(i).name:26s} ctrlrange [{lo:g}, {hi:g}]")
    cam = model.camera("wrist_cam")
    w, h = model.cam_resolution[cam.id]
    fovy = model.cam_fovy[cam.id]
    f = 0.5 * h / np.tan(np.deg2rad(fovy) / 2)
    print(f"  wrist_cam {w}x{h}, fovy {fovy:g} deg -> fx = fy = {f:.2f} px, cx = {w / 2:g}, cy = {h / 2:g}")
    return True


def render(model, data):
    import imageio.v3 as iio

    print("\n== 2. Rendering ==")
    model.vis.quality.offsamples = 0  # exact depth (see check_env.py)
    cam_id = model.camera("wrist_cam").id
    w, h = model.cam_resolution[cam_id]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with mujoco.Renderer(model, height=h, width=w) as r:
        r.update_scene(data, "wrist_cam")
        rgb = r.render()
        r.enable_depth_rendering()
        r.update_scene(data, "wrist_cam")
        depth = r.render()
        r.disable_depth_rendering()
        r.update_scene(data, "overview")
        overview = r.render()

    iio.imwrite(OUT_DIR / "wrist_rgb.png", rgb)
    iio.imwrite(OUT_DIR / "overview.png", overview)
    d_vis = depth.copy()
    d_vis = (255 * (d_vis - d_vis.min()) / (d_vis.max() - d_vis.min())).astype(np.uint8)
    iio.imwrite(OUT_DIR / "wrist_depth.png", d_vis)

    # Depth at the projection of the cube's top-face centre, compared with the true camera-frame z.
    p_world = data.body("cube").xpos + np.array([0, 0, CUBE_HALF])
    R = data.cam_xmat[cam_id].reshape(3, 3)
    p_cam = R.T @ (p_world - data.cam_xpos[cam_id])  # MuJoCo camera frame: -z forward, +y up
    f = 0.5 * h / np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2)
    u = w / 2 + f * p_cam[0] / -p_cam[2]
    v = h / 2 - f * p_cam[1] / -p_cam[2]
    ui, vi = int(round(u)), int(round(v))
    ok = 0 <= ui < w and 0 <= vi < h
    if ok:
        err = abs(depth[vi, ui] - (-p_cam[2]))
        ok = err < 0.005 and rgb[vi, ui, 0] > 150 and rgb[vi, ui, 1] < 80
        print(f"  cube top centre at pixel ({u:.1f}, {v:.1f}), rendered depth {depth[vi, ui]:.4f} m, "
              f"true z {-p_cam[2]:.4f} m, |diff| {1000 * err:.1f} mm, rgb {rgb[vi, ui]}")
    print(f"  cube visible and depth consistent: {'PASS' if ok else 'FAIL'}")
    print(f"  saved images to {OUT_DIR}")
    return ok


def physics(model, data):
    print("\n== 3. Physics (2 s, arm held with qfrc_bias, gripper closes at t = 1 s) ==")
    mujoco.mj_resetDataKeyframe(model, data, model.key("observe").id)
    q0 = data.qpos[:6].copy()
    mujoco.mj_forward(model, data)
    pen = [c.dist for c in data.contact[: data.ncon] if c.dist < -1e-3]
    print(f"  initial contacts: {data.ncon}, penetrations > 1 mm: {len(pen)}")

    grip = model.actuator("gripper_fingers_actuator").id
    while data.time < 2.0:
        data.ctrl[:6] = data.qfrc_bias[:6]
        data.ctrl[grip] = 255 if data.time > 1.0 else 0
        mujoco.mj_step(model, data)

    drift = np.max(np.abs(data.qpos[:6] - q0))
    cube_z = data.body("cube").xpos[2]
    cube_v = np.linalg.norm(data.qvel[model.jnt_dofadr[model.joint("cube_joint").id]:][:6])
    finger = data.qpos[model.jnt_qposadr[model.joint("gripper_right_driver_joint").id]]
    print(f"  arm drift {np.rad2deg(drift):.3f} deg, cube z {cube_z:.4f} m (expect {TABLE_Z + CUBE_HALF:.4f}), "
          f"cube speed {cube_v:.2e}, finger driver {finger:.3f} rad (closed ~0.8)")
    ok = not pen and drift < np.deg2rad(1) and abs(cube_z - TABLE_Z - CUBE_HALF) < 2e-3 and cube_v < 1e-2 \
        and finger > 0.7
    print(f"  stable: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("observe").id)
    mujoco.mj_forward(model, data)

    results = {"model": summary(model), "rendering": render(model, data), "physics": physics(model, data)}
    print("\n== Summary ==")
    for k, v in results.items():
        print(f"  {k:10s} {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
