"""Environment check: library versions, MuJoCo offscreen RGB-D rendering, RTB UR5 dynamics.

Run from the repo root with the venv active:
    python scripts/check_env.py
"""

import os
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = REPO_ROOT / "third_party/mujoco_menagerie/universal_robots_ur5e/scene.xml"
OUT_DIR = REPO_ROOT / "results/check_env"


def check_versions():
    print("== 1. Library versions ==")
    for pkg in ["mujoco", "numpy", "scipy", "opencv-contrib-python", "roboticstoolbox-python"]:
        print(f"  {pkg:24s} {version(pkg)}")
    return True


def check_rendering():
    import imageio.v3 as iio
    import mujoco

    print("\n== 2. MuJoCo offscreen rendering ==")
    print(f"  MUJOCO_GL = {os.environ.get('MUJOCO_GL', '(unset)')}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    # With MSAA on, the resolved depth comes from a single sub-sample offset ~0.4 px from the pixel centre,
    # which adds a few mm of error on surfaces viewed at grazing angles. Disable it for an exact check.
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    height, width = 480, 640
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data)
        rgb = renderer.render()

        renderer.enable_depth_rendering()
        renderer.update_scene(data)
        depth = renderer.render()
        cam = renderer.scene.camera[0]
        cam_pos, cam_fwd = np.array(cam.pos, dtype=float), np.array(cam.forward, dtype=float)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    iio.imwrite(OUT_DIR / "rgb.png", rgb)

    # Pixels that hit nothing (skybox) sit at the far clipping plane; exclude them from the visualisation range.
    zfar = model.vis.map.zfar * model.stat.extent
    hit = depth < 0.99 * zfar
    d_min, d_max = depth[hit].min(), depth[hit].max()
    # Inverse depth keeps nearby structure visible despite the floor stretching to the horizon (near = bright).
    inv = np.where(hit, 1.0 / depth, 0.0)
    vis = (255 * inv / inv.max()).astype(np.uint8)
    iio.imwrite(OUT_DIR / "depth.png", vis)

    print(f"  RGB   {rgb.shape} {rgb.dtype}")
    print(f"  Depth {depth.shape} {depth.dtype}")
    print(f"  Depth min/max (all pixels):     {depth.min():.4f} / {depth.max():.4f}")
    print(f"  Depth min/max (excl. far plane): {d_min:.4f} / {d_max:.4f}   (far plane = {zfar:.2f})")

    # Units check: the centre pixel's depth should equal the geometric distance along the optical axis.
    # mj_ray uses model geometry, not the depth buffer, so agreement means depth is linear and in metres.
    geomgroup = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)  # groups rendered by default
    geomid = np.zeros(1, dtype=np.int32)
    ray_dist = mujoco.mj_ray(model, data, cam_pos, cam_fwd, geomgroup, 1, -1, geomid)
    # With even image dimensions the optical axis passes between the four central pixels.
    centre_depth = depth[height // 2 - 1 : height // 2 + 1, width // 2 - 1 : width // 2 + 1].mean()
    err = abs(centre_depth - ray_dist)
    print(f"  Centre pixel depth: {centre_depth:.4f}   mj_ray distance: {ray_dist:.4f}   |diff| = {err * 1000:.1f} mm")

    ok = ray_dist > 0 and err < 0.001
    print(f"  Depth is linear metric (metres): {'PASS' if ok else 'FAIL'}")
    print(f"  Saved {OUT_DIR / 'rgb.png'} and {OUT_DIR / 'depth.png'}")
    return ok


def check_rtb_dynamics():
    import roboticstoolbox as rtb

    print("\n== 3. Robotics Toolbox UR5 dynamics ==")
    robot = rtb.models.DH.UR5()
    masses = [link.m for link in robot.links]
    for i, link in enumerate(robot.links):
        print(f"  link {i + 1}: m = {link.m:.4f} kg   r = {np.round(link.r, 4)}")
    ok = all(m is not None and m > 0 for m in masses)
    print(f"  Total mass: {sum(masses):.3f} kg")
    print(f"  Dynamic parameters present: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = {
        "versions": check_versions(),
        "rendering": check_rendering(),
        "rtb_dynamics": check_rtb_dynamics(),
    }
    print("\n== Summary ==")
    for name, ok in results.items():
        print(f"  {name:14s} {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
