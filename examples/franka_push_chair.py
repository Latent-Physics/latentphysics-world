"""Franka gives the office chair a ~50 N shove; the chair rolls away on its
casters. IK-scripted force servo — no policies, no learning.

Phases: settle -> IK approach behind the backrest -> creep to gentle contact
-> force-hold shove (feedforward chair velocity + PI force correction around
50 N) until the chair leaves the arm's workspace -> release, chair coasts.

Hard asserts (the clip is a replay of the asserted run):
  mean contact force ~50 N, bounded impact peak, chair displacement >= 0.3 m,
  chair stays upright, casters ROLL (wheel joint travel) rather than slide,
  recline spring loads under the push and returns after, state finite.

Physics runs on the classical C engine (CPU) — this demo is runnable on any
host, no GPU needed; the same MJCF loads unchanged on the LPW GPU engine.

Usage (from examples/):
  python franka_push_chair.py             # headless run + asserts
  python franka_push_chair.py --record    # + docs/media/franka_push_chair.webp
  python franka_push_chair.py --inspect   # chair studio stills to ~/lpw/inspect
  python franka_push_chair.py --viewer    # live on-screen loop
                                          # (macOS: run via mjpython)

Requires a mujoco_menagerie checkout (~/lpw/menagerie or LPW_MENAGERIE) and
the ``demos`` extra (imageio for textures/recording).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402

from _chair import N_CHAIR_ZERO_QPOS, chair_assets, chair_body  # noqa: E402
from _ik import solve_ik  # noqa: E402

MEN = os.path.join(os.environ.get("LPW_MENAGERIE", os.path.expanduser("~/lpw/menagerie")),
                   "franka_emika_panda")
CHAIR_X = 0.72                 # chair root ahead of the panda base
PUSH_Z = 0.60                  # push point height on the backrest shell
TARGET_N = 50.0                # requested push force
DT = 0.005
ARM_HOME = "0 -0.7853 0 -2.3561 0 1.5707 -0.7853"   # retracted ready pose
PUSH_QUAT = np.array([0.7071, 0.0, 0.7071, 0.0])    # gripper z-axis -> +x


def build_scene() -> str:
    """mjx_panda + floor + chair, written into the menagerie panda dir so
    the panda's relative mesh paths resolve (repo pattern)."""
    if not os.path.exists(os.path.join(MEN, "mjx_panda.xml")):
        sys.exit("mujoco_menagerie not found — clone it to ~/lpw/menagerie "
                 "or set LPW_MENAGERIE (see docs/GETTING_STARTED.md)")
    chair = chair_body(pos=(CHAIR_X, 0, 0.004))
    qpos = (f"{ARM_HOME} 0.04 0.04  {CHAIR_X} 0 0.004 1 0 0 0"
            + " 0" * N_CHAIR_ZERO_QPOS)
    xml = f"""<mujoco model="lpw_chair_push">
  <include file="mjx_panda.xml"/>
  <option timestep="{DT}" integrator="implicitfast"/>
  <statistic center="0.9 0 0.6" extent="2.6"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight ambient="0.36 0.36 0.38" diffuse="0.62 0.62 0.62" specular="0.25 0.25 0.25"/>
  </visual>
  <asset>
    {chair_assets()}
  </asset>
  <worldbody>
    <light pos="1.4 -1.6 2.8" dir="-0.35 0.45 -0.8" diffuse="0.75 0.74 0.72" castshadow="true"/>
    <light pos="0.5 1.8 2.4" dir="0.1 -0.6 -0.75" diffuse="0.34 0.35 0.38" castshadow="false"/>
    <geom name="floor" type="plane" size="3.4 3.4 0.1" material="mat_plaster"
          quat="0.981 0 0 0.195" rgba="0.86 0.86 0.87 1"
          friction="0.9 0.005 0.0001"/>
    {chair}
  </worldbody>
  <keyframe>
    <key name="home" qpos="{qpos}" ctrl="{ARM_HOME} 0.0"/>
  </keyframe>
</mujoco>
"""
    path = os.path.join(MEN, "_lpw_chair_push.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


class PushRun:
    """One deterministic push episode on a compiled scene."""

    def __init__(self, m):
        self.m = m
        self.d = mujoco.MjData(m)
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        assert self.site >= 0
        self.chair_bodies = {i for i in range(m.nbody)
                             if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "")
                             .startswith(("chair", "caster", "wheel"))}
        self._ikd = mujoco.MjData(m)
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "chair_free")
        self.chair_qadr = m.jnt_qposadr[j]
        self.chair_vadr = m.jnt_dofadr[j]
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "chair_recline")
        self.rec_adr = m.jnt_qposadr[j]
        self.wheel_adr = [m.jnt_qposadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"wheel{i}_roll")] for i in range(5)]

    def chair_push_force_x(self):
        """World +x force the robot applies to the chair (sum over contacts)."""
        fx = 0.0
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1 = self.m.geom_bodyid[c.geom1]
            b2 = self.m.geom_bodyid[c.geom2]
            in1, in2 = b1 in self.chair_bodies, b2 in self.chair_bodies
            if in1 == in2 or 0 in (b1, b2):
                continue                      # not a robot-chair contact
            mujoco.mj_contactForce(self.m, self.d, i, f6)
            frame = c.frame.reshape(3, 3)     # rows: normal, tan1, tan2
            fw = f6[0] * frame[0] + f6[1] * frame[1] + f6[2] * frame[2]
            fx += fw[0] if in2 else -fw[0]    # mj_contactForce acts on geom2
        return fx

    def chair_pos(self):
        return self.d.qpos[self.chair_qadr:self.chair_qadr + 3].copy()

    def chair_tilt_deg(self):
        q = self.d.qpos[self.chair_qadr + 3:self.chair_qadr + 7]
        return np.degrees(2 * np.arccos(min(1.0, abs(q[0]))))

    def _servo_arm(self, target_pos, seed):
        q = solve_ik(self.m, self._ikd, self.site, target_pos, PUSH_QUAT, seed)
        self.d.ctrl[:7] = q
        return q

    def run(self, step_cb=None):
        """Run one episode; returns (peak_force, trace, start_pos, rec0)."""
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.ctrl[7] = 0.0                      # fingers closed: knuckle-push
        seed = d.qpos[:7].copy()
        trace = {"t": [], "fx": [], "chair_x": [], "rec": [], "qpos": []}
        peak = 0.0

        def step(n, phase):
            nonlocal peak
            for _ in range(n):
                mujoco.mj_step(m, d)
                fx = self.chair_push_force_x()
                if phase == "push":          # peak = push-phase force only
                    peak = max(peak, fx)
                trace["t"].append(d.time)
                trace["fx"].append(fx)
                trace["chair_x"].append(d.qpos[self.chair_qadr])
                trace["rec"].append(d.qpos[self.rec_adr])
                trace["qpos"].append(d.qpos.copy())
                if step_cb:
                    step_cb(d, phase)

        step(80, "settle")
        start = self.chair_pos()
        rec0 = d.qpos[self.rec_adr]          # recline equilibrium under gravity
        # approach behind the backrest (IK waypoints, continuous seeding)
        p0 = np.array([0.365, 0.0, PUSH_Z])
        home_p = d.site_xpos[self.site].copy()
        for k in (0.4, 0.75, 1.0):
            seed = self._servo_arm(home_p + k * (p0 - home_p), seed)
            step(60, "approach")
        # creep in until first contact (gentle touch, no impact spike)
        tgt = p0.copy()
        for _ in range(int(1.8 / DT)):
            tgt[0] += 0.03 * DT
            seed = self._servo_arm(tgt, seed)
            step(1, "touch")
            if trace["fx"][-1] > 2.0:
                break
        # force-hold shove: feedforward the chair's velocity plus a PI force
        # correction, so the hand tracks the accelerating chair at ~50 N
        tgt[0] += 0.006                      # initial bite: skip the soft ramp
        f_int = 0.0
        for _ in range(int(2.0 / DT)):
            err = TARGET_N - trace["fx"][-1]
            f_int = np.clip(f_int + err * DT, -0.4, 0.4)
            v_cmd = min(max(0.0, d.qvel[self.chair_vadr] + 0.003 * err + 0.08 * f_int), 2.5)
            tgt[0] += v_cmd * DT
            seed = self._servo_arm(tgt, seed)
            step(1, "push")
            if (d.site_xpos[self.site][0] > 0.78
                    or d.qpos[self.chair_qadr] - start[0] > 0.42):
                break
        # release and let it roll
        seed = self._servo_arm(np.array([0.30, 0.0, PUSH_Z]), seed)
        step(100, "release")
        d.ctrl[:7] = np.fromstring(ARM_HOME, sep=" ")
        step(int(2.2 / DT) - 100, "roll")
        return peak, trace, start, rec0


def check(runner, peak, trace, start, rec0):
    """The hard gate every run (and therefore every clip) must pass."""
    d = runner.d
    fx = np.array(trace["fx"])
    f_hold = fx[fx > 10.0]
    hold_mean = float(f_hold.mean()) if len(f_hold) else 0.0
    hold_dur = float(len(f_hold) * DT)
    disp = runner.chair_pos() - start
    tilt = runner.chair_tilt_deg()
    wheel_travel = float(np.mean([abs(d.qpos[a]) for a in runner.wheel_adr]))
    rec_defl = max(trace["rec"]) - rec0

    print(f"push force: mean {hold_mean:.1f} N over {hold_dur:.2f} s of contact, "
          f"peak {peak:.1f} N (target {TARGET_N:.0f} N)")
    print(f"chair displacement dx={disp[0]:+.3f} m | tilt {tilt:.2f} deg | "
          f"wheel travel {wheel_travel:.1f} rad | recline deflection {rec_defl:.4f} rad")

    assert np.isfinite(d.qpos).all(), "non-finite state"
    assert 40.0 <= hold_mean <= 58.0, f"mean push force {hold_mean:.1f} N not ~50 N"
    assert hold_dur >= 0.25, f"contact too brief: {hold_dur:.2f} s"
    assert 45.0 <= peak <= 95.0, f"push peak {peak:.1f} N out of range"
    assert disp[0] >= 0.30, f"chair only moved {disp[0]:.3f} m"
    assert tilt < 10.0, f"chair tipped: {tilt:.1f} deg"
    assert wheel_travel > 2.0, f"casters slid instead of rolling ({wheel_travel:.1f} rad)"
    assert rec_defl > 0.008, f"recline spring never loaded ({rec_defl:.4f} rad)"
    assert abs(d.qpos[runner.rec_adr] - rec0) < 0.03, "recline spring did not return"
    print("ALL ASSERTS PASSED")


CAM = dict(lookat=(0.85, 0, 0.50), distance=2.9, azimuth=118, elevation=-16)


def run_viewer(scene_path):
    """Loop the push episode live in the interactive viewer, in real time.
    On macOS this must be run via mjpython."""
    import time

    import mujoco.viewer

    m = mujoco.MjModel.from_xml_path(scene_path)
    runner = PushRun(m)
    try:
        handle = mujoco.viewer.launch_passive(runner.m, runner.d)
    except RuntimeError as e:
        sys.exit(f"viewer failed to launch ({e}).\n"
                 "On macOS the interactive viewer needs mjpython:\n"
                 "  mjpython examples/franka_push_chair.py --viewer")
    with handle as viewer:
        viewer.cam.azimuth, viewer.cam.elevation = CAM["azimuth"], CAM["elevation"]
        viewer.cam.distance = CAM["distance"]
        viewer.cam.lookat[:] = CAM["lookat"]
        clock = {"next": time.monotonic()}

        def pace(d, phase):
            if not viewer.is_running():
                raise SystemExit
            viewer.sync()
            clock["next"] += DT
            lag = clock["next"] - time.monotonic()
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.5:                 # fell behind: resync, don't rush
                clock["next"] = time.monotonic()

        episode = 0
        while viewer.is_running():
            episode += 1
            print(f"episode {episode} (close the window to stop)")
            try:
                peak, trace, start, rec0 = runner.run(step_cb=pace)
                check(runner, peak, trace, start, rec0)
            except SystemExit:
                break
            for _ in range(90):              # hold the final frame ~1.5 s
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(1 / 60)
            clock["next"] = time.monotonic()


def run_inspect():
    """Charter-style look pass: studio stills, neutral + articulated."""
    from PIL import Image

    from _chair import chair_scene_xml

    out_dir = os.path.expanduser("~/lpw/inspect")
    os.makedirs(out_dir, exist_ok=True)
    m = mujoco.MjModel.from_xml_string(chair_scene_xml())
    d = mujoco.MjData(m)
    poses = {"neutral": {}, "articulated": {
        "chair_swivel": np.radians(35), "chair_recline": 0.20,
        "chair_head_pitch": 0.15, "chair_lift": 0.035,
        "chair_head_slide": 0.03, "chair_arm_l_slide": 0.05,
        "chair_arm_r_slide": 0.05,
        **{f"caster{i}_swivel": np.radians(30 + 55 * i) for i in range(5)}}}
    r = mujoco.Renderer(m, height=1440, width=1920)
    cam = mujoco.MjvCamera()
    for pose, joints in poses.items():
        mujoco.mj_resetData(m, d)
        for name, val in joints.items():
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            d.qpos[m.jnt_qposadr[j]] = val
        mujoco.mj_forward(m, d)
        for view, (az, el, dist, look) in {
                "hero": (145, -14, 2.35, (0.02, 0, 0.62)),
                "side": (90, -10, 2.5, (0, 0, 0.65))}.items():
            cam.azimuth, cam.elevation, cam.distance = az, el, dist
            cam.lookat = np.array(look)
            r.update_scene(d, camera=cam)
            img = Image.fromarray(r.render()).resize((960, 720), Image.LANCZOS)
            img.save(os.path.join(out_dir, f"chair_{pose}_{view}.png"))
    r.close()
    print(f"stills -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true",
                    help="replay the asserted run to docs/media/franka_push_chair.webp")
    ap.add_argument("--inspect", action="store_true",
                    help="render chair studio stills (neutral + articulated)")
    ap.add_argument("--viewer", action="store_true",
                    help="loop the push live in the interactive viewer")
    args = ap.parse_args()

    if args.inspect:
        run_inspect()
        return
    scene = build_scene()
    if args.viewer:
        run_viewer(scene)
        return

    m = mujoco.MjModel.from_xml_path(scene)
    runner = PushRun(m)
    peak, trace, start, rec0 = runner.run()
    check(runner, peak, trace, start, rec0)

    if args.record:
        from _record import record_webp
        record_webp(scene, trace["qpos"], "franka_push_chair",
                    cam=CAM, every=4, fps=15, size=(640, 400))


if __name__ == "__main__":
    main()
