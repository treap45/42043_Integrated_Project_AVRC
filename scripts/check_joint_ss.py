"""Check src/model/joint_ss.py: discretisation, caching and rank conditions.

Run from the repo root with the venv active:
    python scripts/check_joint_ss.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.joint_ss import build_joint_models, controllability_matrix, observability_matrix  # noqa: E402
from src.model.ur3e import DEFAULT_XML, UR3eModel  # noqa: E402

DT = 0.002


def main():
    robot = UR3eModel()
    q_op = mujoco.MjModel.from_xml_path(str(DEFAULT_XML)).key("observe").qpos.copy()
    models = build_joint_models(robot, q_op, DT)

    ok = True
    for i, js in enumerate(models):
        Ad, Bd = js.discretize()
        Ad_c, Bd_c = js.discretize_closed_form()
        zoh_ok = np.allclose(Ad, Ad_c, rtol=1e-10, atol=1e-12) and np.allclose(Bd, Bd_c, rtol=1e-10, atol=1e-12)
        cached_ok = js._zoh is js._zoh and js.Ad is not js.Ad  # computed once, copies handed out
        ranks = (np.linalg.matrix_rank(controllability_matrix(js.A, js.B)),
                 np.linalg.matrix_rank(controllability_matrix(Ad, Bd)),
                 np.linalg.matrix_rank(observability_matrix(js.A, js.C)),
                 np.linalg.matrix_rank(observability_matrix(Ad, js.C)))
        joint_ok = zoh_ok and cached_ok and ranks == (2, 2, 3, 3)
        ok &= joint_ok
        print(f"  joint {i}: J = {js.inertia:.4f}  expm == closed form {zoh_ok}  cached {cached_ok}  "
              f"rank ctrb c/d {ranks[0]}/{ranks[1]}  obsv c/d {ranks[2]}/{ranks[3]}  {'PASS' if joint_ok else 'FAIL'}")
    print(f"\n  joint_ss  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
