#!/usr/bin/env bash
# Runs every simulation experiment reported on the project page and writes one log per experiment.
# Takes about an hour on 16 cores. Run from the repository root with the virtual environment active:
#
#     bash scripts/evaluate_all.sh                # results/eval/<name>.log
#     JOBS=4 bash scripts/evaluate_all.sh         # fewer at a time
#
# The occluded-level experiments (--occluders) need the code on the `advanced` branch; they are skipped elsewhere.
# scripts/calibrate_camera.py and the lab pipeline in simulation run separately, since they write configs/ and one
# directory per run: python scripts/run_real.py --sim ... and python scripts/lab_sim_fast.py --sim ...
# The logs behind the page and the script that builds its tables and charts are on branch `github_page`.
set -u
OUT=results/eval
JOBS=${JOBS:-16}
mkdir -p "$OUT"
export MUJOCO_GL=${MUJOCO_GL:-egl} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PAUSE="--motion long_pause --speed 0.02 0.03 --duration 10 12"
{
cat <<'EOF'
moving_lqr|main.py --no-window --trials 100
moving_pd|main.py --no-window --trials 20 --mode pd
moving_fixed|main.py --no-window --trials 20 --fixed-gains
moving_noiseless|main.py --no-window --trials 20 --no-camera-noise
moving_nominal|main.py --no-window --trials 20 --nominal-camera
moving_ct|main.py --no-window --trials 20 --cube-filter ct
static_open|main.py --no-window --trials 100 --no-moving-cube --approach open
static_pbvs|main.py --no-window --trials 100 --no-moving-cube --approach pbvs
static_pd|main.py --no-window --trials 20 --no-moving-cube --approach open --mode pd
push_open|main.py --no-window --trials 20 --no-moving-cube --approach open --push-cube 0.025
push_pbvs|main.py --no-window --trials 20 --no-moving-cube --approach pbvs --push-cube 0.025
push_open15|main.py --no-window --trials 20 --no-moving-cube --approach open --push-cube 0.015
push_pbvs15|main.py --no-window --trials 20 --no-moving-cube --approach pbvs --push-cube 0.015
he2_open|main.py --no-window --trials 20 --no-moving-cube --approach open --handeye-error-deg 2
he2_pbvs|main.py --no-window --trials 20 --no-moving-cube --approach pbvs --handeye-error-deg 2
he2_track|main.py --no-window --trials 20 --handeye-error-deg 2
filters|scripts/compare_cube_filters.py --seeds 20
filters_r|scripts/compare_cube_filters.py --seeds 20 --motions line --r-scales 0.1 1 10
motion_line_cv|scripts/run_cube_motion_trials.py --motion line --filter cv
motion_line_ct|scripts/run_cube_motion_trials.py --motion line --filter ct
motion_arc_cv|scripts/run_cube_motion_trials.py --motion arc --filter cv
motion_arc_ct|scripts/run_cube_motion_trials.py --motion arc --filter ct
motion_s_cv|scripts/run_cube_motion_trials.py --motion s_curve --filter cv
motion_s_ct|scripts/run_cube_motion_trials.py --motion s_curve --filter ct
motion_var_cv|scripts/run_cube_motion_trials.py --motion variable --filter cv --turn-radius 0.25 0.40
motion_var_ct|scripts/run_cube_motion_trials.py --motion variable --filter ct --turn-radius 0.25 0.40
plot_ekf|scripts/plot_cube_ekf.py --seed 0
chk_joint_ss|scripts/check_joint_ss.py
chk_joint_kf|scripts/check_joint_kf.py
chk_control_lqr|scripts/check_control.py
chk_control_pd|scripts/check_control.py --mode pd
chk_ctrl_model|scripts/check_controller_model.py
chk_sim|scripts/check_sim.py
chk_noise|scripts/check_camera_noise.py
chk_cube_ekf|scripts/check_cube_ekf.py
chk_cube_ekf_nominal|scripts/check_cube_ekf.py --nominal-camera --seeds 20
chk_cube_motion|scripts/check_cube_motion.py
chk_eye_to_hand|scripts/check_eye_to_hand.py
EOF
for pause in 1.0 1.5 2.0 2.5 3.0 4.0; do
  echo "pause_$pause|scripts/run_cube_motion_trials.py $PAUSE --pause $pause"
done
if grep -q -- "--occluders" main.py; then
  echo "l1_off|main.py --no-window --trials 50 --level L1 --no-moving-cube"
  echo "l1_conservative|main.py --no-window --trials 50 --level L1 --no-moving-cube --occluders conservative"
  echo "l1_active|main.py --no-window --trials 50 --level L1 --no-moving-cube --occluders active"
  echo "l0_active|main.py --no-window --trials 20 --occluders active"
fi
} | xargs -d '\n' -P "$JOBS" -I{} bash -c '
  IFS="|" read -r name cmd <<<"$1"
  python $cmd > "'"$OUT"'/$name.log" 2>&1
  echo "$name exit $?"
' _ {}
