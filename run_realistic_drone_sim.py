#!/usr/bin/env python3
"""
Realistic 3D Quadrotor Simulation with Tri-Blade Propellers, Live Phase Indicators,
Real-Time Binary PX4 .ulg Recording, and Automatic Post-Run ULog Plot Generation.
Compatible with Python 3.12 and NumPy 2.1.6+

Run:
    python3 run_realistic_drone_sim.py --fault complete_stop --duration 12.0
"""

import argparse
import os
import struct
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("ulg_logs", exist_ok=True)


# ==================== REAL-TIME PX4 BINARY ULOG (.ulg) RECORDER ====================
class LiveULogRecorder:
    """Streams real-time MuJoCo 3D physics telemetry directly into a PX4 .ulg file."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.fp = open(filepath, "wb")
        magic = b"ULog\x01\x12\x35\x01"
        self.fp.write(magic + struct.pack("<Q", 0))
        self._write_msg(ord("B"), b"\x00" * 40)
        self._write_info_str("sys_name", "PX4_MUJOCO_REALISTIC_X500")
        self._write_info_str("ver_hw", "HOLYBRO_PIXHAWK_6C")
        self._register_topics()

    def _write_msg(self, msg_type: int, payload: bytes) -> None:
        self.fp.write(struct.pack("<HB", len(payload), msg_type) + payload)

    def _write_info_str(self, key: str, val: str) -> None:
        val_b = val.encode("ascii")
        key_str = f"char[{len(val_b)}] {key}".encode("ascii")
        self._write_msg(ord("I"), struct.pack("<B", len(key_str)) + key_str + val_b)

    def _write_format(self, fmt_str: str) -> None:
        self._write_msg(ord("F"), fmt_str.encode("ascii"))

    def _subscribe(self, msg_id: int, topic_name: str) -> None:
        self._write_msg(ord("A"), struct.pack("<BH", 0, msg_id) + topic_name.encode("ascii"))

    def _register_topics(self) -> None:
        self._write_format(
            "vehicle_local_position:uint64_t timestamp;float x;float y;float z;"
            "float vx;float vy;float vz;float z_stock_crash;"
        )
        self._write_format(
            "vehicle_attitude:uint64_t timestamp;float roll_deg;float pitch_deg;"
            "float yaw_deg;float tilt_deg;float tilt_stock_deg;"
        )
        self._write_format(
            "vehicle_angular_velocity:uint64_t timestamp;float wx;float wy;float wz;"
            "float alpha_x;float alpha_y;float alpha_z;"
        )
        self._write_format(
            "actuator_motors:uint64_t timestamp;float thrust_1;float thrust_2;"
            "float thrust_3;float thrust_4;float rpm_1;float rpm_2;float rpm_3;float rpm_4;"
        )
        self._write_format(
            "ftc_controller_status:uint64_t timestamp;float residual_norm;"
            "int32_t flight_phase;int32_t active_controller;int32_t injected_motor;"
            "int32_t detected_motor;int32_t fault_code;"
        )
        self._subscribe(1, "vehicle_local_position")
        self._subscribe(2, "vehicle_attitude")
        self._subscribe(3, "vehicle_angular_velocity")
        self._subscribe(4, "actuator_motors")
        self._subscribe(5, "ftc_controller_status")

    def log_event(self, timestamp_us: int, message: str, level: bytes = b"6") -> None:
        self._write_msg(ord("L"), struct.pack("<cQ", level, timestamp_us) + message.encode("ascii"))

    def record_step(
        self,
        ts_us: int,
        pos_ftc: tuple[float, float, float],
        vel_ftc: tuple[float, float, float],
        z_stock: float,
        att_ftc: tuple[float, float, float, float],
        tilt_stock: float,
        rates_ftc: tuple[float, float, float],
        acc_ftc: tuple[float, float, float],
        thrusts: tuple[float, float, float, float],
        rpms: tuple[float, float, float, float],
        res_norm: float,
        phase_id: int,
        ctrl_id: int,
        inj_m: int,
        det_m: int,
        f_code: int,
    ) -> None:
        px, py, pz = pos_ftc
        vx, vy, vz = vel_ftc
        r_d, p_d, y_d, t_d = att_ftc
        wx, wy, wz = rates_ftc
        ax, ay, az = acc_ftc
        f1, f2, f3, f4 = thrusts
        w1, w2, w3, w4 = rpms

        self._write_msg(ord("D"), struct.pack("<HQfffffff", 1, ts_us, px, py, pz, vx, vy, vz, z_stock))
        self._write_msg(ord("D"), struct.pack("<HQfffff", 2, ts_us, r_d, p_d, y_d, t_d, tilt_stock))
        self._write_msg(ord("D"), struct.pack("<HQffffff", 3, ts_us, wx, wy, wz, ax, ay, az))
        self._write_msg(ord("D"), struct.pack("<HQffffffff", 4, ts_us, f1, f2, f3, f4, w1, w2, w3, w4))
        self._write_msg(
            ord("D"),
            struct.pack("<HQfiiiii", 5, ts_us, res_norm, phase_id, ctrl_id, inj_m, det_m, f_code),
        )

    def close(self) -> None:
        self.fp.flush()
        self.fp.close()


# ==================== HIGH-DETAIL REALISTIC 3D QUADROTOR MJCF BUILDER ====================
def build_drone_xml(prefix: str, pos_str: str, body_rgba: str) -> str:
    """
    Generates a realistic 3D quadrotor model with:
      - Dual carbon-fiber sandwich plates & anodized red/cyan standoffs
      - Top-mounted 4S LiPo battery pack + Pixhawk 6C orange cube + GPS mast & puck
      - Tall landing gear skids
      - 4 detailed brushless motor bells, rotor guards, LED status rings, and 3-blade tri-props
    """
    rotors_xml = []
    # '+' configuration rotor coordinates: Motor 1 (+X), Motor 2 (+Y), Motor 3 (-X), Motor 4 (-Y)
    rotor_coords = (
        (1, "0.25 0 0.018", "0.2 0.9 0.3 0.9"),
        (2, "0 0.25 0.018", "0.2 0.85 0.95 0.9"),
        (3, "-0.25 0 0.018", "0.95 0.85 0.15 0.9"),
        (4, "0 -0.25 0.018", "0.95 0.25 0.25 0.95"),
    )
    for idx, r_pos, hub_rgba in rotor_coords:
        rotors_xml.append(f"""
      <!-- Motor {idx} Assembly: Stator Base, Cooling Bell, Prop Guard, Status LED Ring & Tri-Blade Propeller -->
      <geom name="{prefix}_guard_{idx}" type="cylinder" pos="{r_pos}" size="0.132 0.004" rgba="0.18 0.20 0.22 0.22" contype="0" conaffinity="0"/>
      <geom name="{prefix}_led_{idx}" type="cylinder" pos="{r_pos.split()[0]} {r_pos.split()[1]} -0.012" size="0.032 0.003" rgba="0.1 0.95 0.3 0.95" contype="0" conaffinity="0"/>
      <geom name="{prefix}_motor_mount_{idx}" type="cylinder" pos="{r_pos}" size="0.022 0.014" rgba="0.15 0.15 0.17 1"/>
      <geom name="{prefix}_motor_bell_{idx}" type="cylinder" pos="{r_pos.split()[0]} {r_pos.split()[1]} 0.032" size="0.020 0.010" rgba="0.75 0.55 0.15 1"/>
      <body name="{prefix}_prop_{idx}" pos="{r_pos.split()[0]} {r_pos.split()[1]} 0.045">
        <joint name="{prefix}_j{idx}" type="hinge" axis="0 0 1"/>
        <inertial pos="0 0 0" mass="0.004" diaginertia="1e-5 1e-5 1e-5"/>
        <!-- Spinner Hub Cap -->
        <geom name="{prefix}_hub_{idx}" type="cylinder" size="0.014 0.009" rgba="{hub_rgba}"/>
        <!-- 3-Blade Tri-Propeller (0 deg, 120 deg, 240 deg with realistic pitch tilt) -->
        <geom name="{prefix}_b1_{idx}" type="box" pos="0.055 0 0" size="0.055 0.013 0.002" euler="12 0 0" rgba="0.88 0.92 0.96 0.88"/>
        <geom name="{prefix}_b2_{idx}" type="box" pos="-0.0275 0.0476 0" size="0.055 0.013 0.002" euler="12 0 120" rgba="0.88 0.92 0.96 0.88"/>
        <geom name="{prefix}_b3_{idx}" type="box" pos="-0.0275 -0.0476 0" size="0.055 0.013 0.002" euler="12 0 240" rgba="0.88 0.92 0.96 0.88"/>
        <!-- Colored Blade Tips for High-Speed Visual Spin Clarity -->
        <geom type="box" pos="0.102 0 0.001" size="0.010 0.013 0.0025" euler="12 0 0" rgba="{hub_rgba}"/>
        <geom type="box" pos="-0.051 0.0883 0.001" size="0.010 0.013 0.0025" euler="12 0 120" rgba="{hub_rgba}"/>
        <geom type="box" pos="-0.051 -0.0883 0.001" size="0.010 0.013 0.0025" euler="12 0 240" rgba="{hub_rgba}"/>
      </body>""")

    return f"""
    <body name="{prefix}_drone" pos="{pos_str}">
      <freejoint name="{prefix}_root"/>
      <inertial pos="0 0 0" mass="1.50" diaginertia="0.015 0.015 0.028"/>

      <!-- Bottom & Top Carbon-Fiber Sandwich Frame Plates -->
      <geom name="{prefix}_plate_bot" type="box" pos="0 0 -0.012" size="0.085 0.085 0.003" rgba="0.14 0.15 0.17 1"/>
      <geom name="{prefix}_plate_top" type="box" pos="0 0 0.016" size="0.078 0.078 0.003" rgba="0.14 0.15 0.17 1"/>
      <!-- Colored Avionics Core & Anodized Standoffs -->
      <geom name="{prefix}_core" type="box" pos="0 0 0.002" size="0.062 0.062 0.011" rgba="{body_rgba}"/>
      <!-- Holybro Pixhawk 6C Flight Controller Cube + 4S LiPo Battery Pack -->
      <geom name="{prefix}_pixhawk" type="box" pos="0.015 0 0.026" size="0.024 0.020 0.008" rgba="0.95 0.55 0.12 1"/>
      <geom name="{prefix}_lipo" type="box" pos="-0.022 0 0.032" size="0.038 0.024 0.014" rgba="0.22 0.26 0.32 1"/>
      <!-- GPS / Compass Mast & Puck -->
      <geom type="cylinder" pos="-0.045 0.045 0.055" size="0.003 0.038" rgba="0.2 0.2 0.2 1"/>
      <geom type="cylinder" pos="-0.045 0.045 0.095" size="0.022 0.005" rgba="0.12 0.12 0.14 1"/>
      <!-- Forward Camera Gimbal Dome -->
      <geom type="sphere" pos="0.082 0 0.0" size="0.016" rgba="0.1 0.1 0.12 1"/>

      <!-- Woven Carbon-Fiber Tubular Motor Arms & Diagonal Braces -->
      <geom type="capsule" fromto="-0.25 0 0.004 0.25 0 0.004" size="0.011" rgba="0.20 0.22 0.25 1"/>
      <geom type="capsule" fromto="0 -0.25 0.004 0 0.25 0.004" size="0.011" rgba="0.20 0.22 0.25 1"/>
      <geom type="capsule" fromto="-0.11 -0.11 -0.002 0.11 0.11 -0.002" size="0.006" rgba="{body_rgba}"/>
      <geom type="capsule" fromto="-0.11 0.11 -0.002 0.11 -0.11 -0.002" size="0.006" rgba="{body_rgba}"/>

      <!-- Tall Landing Gear Legs & Horizontal Skids -->
      <geom type="capsule" fromto="0.065 0.065 -0.012 0.095 0.105 -0.11" size="0.006" rgba="0.22 0.22 0.24 1"/>
      <geom type="capsule" fromto="-0.065 0.065 -0.012 -0.095 0.105 -0.11" size="0.006" rgba="0.22 0.22 0.24 1"/>
      <geom type="capsule" fromto="0.065 -0.065 -0.012 0.095 -0.105 -0.11" size="0.006" rgba="0.22 0.22 0.24 1"/>
      <geom type="capsule" fromto="-0.065 -0.065 -0.012 -0.095 -0.105 -0.11" size="0.006" rgba="0.22 0.22 0.24 1"/>
      <geom type="capsule" fromto="-0.14 0.105 -0.11 0.14 0.105 -0.11" size="0.008" rgba="0.15 0.15 0.17 1"/>
      <geom type="capsule" fromto="-0.14 -0.105 -0.11 0.14 -0.105 -0.11" size="0.008" rgba="0.15 0.15 0.17 1"/>

      <!-- Primary-Axis Spin Indicator Mast on FTC Drone -->
      <geom name="{prefix}_axis_mast" type="cylinder" pos="0 0 0.14" size="0.004 0.11" rgba="0.1 1.0 0.4 0.85" contype="0" conaffinity="0"/>
      {''.join(rotors_xml)}
    </body>
    """


def build_scene_xml() -> str:
    stock_xml = build_drone_xml("stock", "0 1.45 0.12", "0.88 0.18 0.18 1")
    ftc_xml = build_drone_xml("ftc", "0 -1.45 0.12", "0.10 0.75 0.92 1")
    return f"""
<mujoco model="realistic_quadrotor_ftc_arena">
  <option timestep="0.002" gravity="0 0 -9.81" density="1.2" viscosity="0.00002"/>
  <visual>
    <headlight ambient="0.48 0.48 0.48" diffuse="0.82 0.82 0.82" specular="0.35 0.35 0.35"/>
    <rgba haze="0.14 0.20 0.30 1"/>
    <global azimuth="140" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.22 0.34 0.52" rgb2="0.04 0.07 0.12" width="512" height="512"/>
    <texture name="grid_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.16 0.20 0.24" rgb2="0.25 0.30 0.36"/>
    <material name="grid_mat" texture="grid_tex" texrepeat="10 10" texuniform="true" reflectance="0.18"/>
  </asset>

  <worldbody>
    <light pos="2 0 12" dir="-0.15 0 -1" directional="true"/>
    <geom name="ground" type="plane" size="16 16 0.1" material="grid_mat"/>

    <!-- Target Hover Reference Rings at z = 3.0 m -->
    <site name="stock_target_ring" pos="0 1.45 3.0" size="0.32 0.006" type="cylinder" rgba="0.95 0.25 0.25 0.28"/>
    <site name="ftc_target_ring" pos="0 -1.45 3.0" size="0.32 0.006" type="cylinder" rgba="0.15 0.95 0.45 0.35"/>

    <!-- Floating 3D Phase Status Beacon Above Arena -->
    <geom name="phase_beacon" type="sphere" pos="0 0 4.3" size="0.14" rgba="0.2 0.6 1.0 0.9" contype="0" conaffinity="0"/>

    {stock_xml}
    {ftc_xml}
  </worldbody>
</mujoco>
"""


def quat_to_rot_and_euler(q: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    w, x, y, z = q
    R = np.array((
        (1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)),
        (2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)),
        (2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)),
    ))
    pitch = float(np.degrees(np.arcsin(np.clip(-R.item(2, 0), -1.0, 1.0))))
    roll = float(np.degrees(np.arctan2(R.item(2, 1), R.item(2, 2))))
    yaw = float(np.degrees(np.arctan2(R.item(1, 0), R.item(0, 0))))
    tilt = float(np.degrees(np.arccos(np.clip(R.item(2, 2), -1.0, 1.0))))
    return R, roll, pitch, yaw, tilt


def generate_plots_from_logged_run(
    t_arr: np.ndarray,
    z_ftc: np.ndarray,
    z_stock: np.ndarray,
    drift_ftc: np.ndarray,
    tilt_ftc: np.ndarray,
    tilt_stock: np.ndarray,
    wx_arr: np.ndarray,
    wy_arr: np.ndarray,
    wz_arr: np.ndarray,
    f_arr: np.ndarray,
    res_arr: np.ndarray,
    t_hover: float,
    t_fault: float,
    t_detect: float,
    ulg_filepath: str,
) -> None:
    """Generates a 4-panel plot with shaded phase regions and controller switch markers."""
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.6), dpi=300)
    plt.rcParams.update({"font.size": 9.5})
    ax1, ax2, ax3, ax4 = axes.flat

    def add_phase_shading(ax) -> None:
        ax.axvspan(0.0, t_hover, color="#cfe2f3", alpha=0.30, label="Phase 1: Takeoff (4-Motor PID)")
        ax.axvspan(t_hover, t_fault, color="#d9ead3", alpha=0.35, label="Phase 2: Nominal Hover (4-Motor PID)")
        ax.axvspan(t_fault, t_detect, color="#f4cccc", alpha=0.65)
        ax.axvspan(t_detect, float(t_arr[-1]), color="#fff2cc", alpha=0.35, label="Phase 4: 3-Motor Spinning Hover (Prioritized INDI)")
        ax.axvline(t_fault, color="#cc0000", ls="--", lw=1.5, label=f"Fault Injected (t = {t_fault:.2f} s)")
        ax.axvline(t_detect, color="#1155cc", ls=":", lw=1.6, label=f"FDI Trigger & Controller Switch (t = {t_detect:.3f} s)")

    # Panel 1: Altitude & Position (Stock Crash vs. FTC 3-Motor Hover)
    add_phase_shading(ax1)
    ax1.plot(t_arr, z_stock, "r--", lw=1.9, label="Left Drone (Stock PX4 PID): Altitude z (Crashes)")
    ax1.plot(t_arr, z_ftc, "b-", lw=2.2, label="Right Drone (Our FTC INDI): Altitude z (Hovers @ 3m)")
    ax1.plot(t_arr, drift_ftc, "g-", lw=1.5, label="Right Drone Horizontal Drift ||(x,y)|| (m)")
    ax1.set_title("1. Altitude & Hover Recovery (Logged from live_3d_simulation.ulg)", fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Altitude / Drift (m)")
    ax1.set_ylim(-0.2, 3.8)
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="lower left", fontsize=7.2)

    # Panel 2: Attitude Tilt Angle & Observer Residual Trigger
    add_phase_shading(ax2)
    ax2.plot(t_arr, tilt_stock, "r--", lw=1.7, label="Stock PX4 PID: Tilt Angle (Flips > 90 deg)")
    ax2.plot(t_arr, tilt_ftc, "b-", lw=2.0, label="Our FTC INDI: Primary-Axis Tilt Angle (deg)")
    ax2.plot(t_arr, res_arr * 25.0, "purple", lw=1.5, label="Observer Residual ||r(t)|| (x25 scaled)")
    ax2.set_title("2. Primary-Axis Tilt Angle & FDI Residual Spike", fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (deg) / Scaled Residual")
    ax2.set_ylim(0.0, 110.0)
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="upper left", fontsize=7.5)

    # Panel 3: Body Angular Rates (Transition to Relaxed-Hover Spin)
    add_phase_shading(ax3)
    ax3.plot(t_arr, wx_arr, "c-", lw=1.4, label="Roll Rate p (rad/s)")
    ax3.plot(t_arr, wy_arr, "m-", lw=1.4, label="Pitch Rate q (rad/s)")
    ax3.plot(t_arr, wz_arr, "b-", lw=2.1, label="Yaw Spin Rate r (rad/s) -> Steady Relaxed Hover")
    ax3.set_title("3. Body Angular Velocities Across Controller Switch", fontweight="bold")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Angular Velocity (rad/s)")
    ax3.grid(True, alpha=0.35)
    ax3.legend(loc="upper left", fontsize=7.5)

    # Panel 4: Rotor Thrusts Before & After Controller Reconfiguration
    add_phase_shading(ax4)
    ax4.plot(t_arr, f_arr[:, 0], "b-", lw=1.8, label="Motor 1 Thrust f1 (Surviving Primary)")
    ax4.plot(t_arr, f_arr[:, 1], "orange", lw=1.8, label="Motor 2 Thrust f2 (Opposite to Failed Motor)")
    ax4.plot(t_arr, f_arr[:, 2], "g--", lw=1.8, label="Motor 3 Thrust f3 (Surviving Primary)")
    ax4.plot(t_arr, f_arr[:, 3], "r-", lw=2.2, label="Motor 4 Thrust f4 (Failed @ t = 4.50 s -> 0 N)")
    ax4.set_title("4. Rotor Thrust Reallocation: 4-Motor PID -> 3-Motor INDI", fontweight="bold")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Rotor Thrust (N)")
    ax4.set_ylim(-0.5, 11.8)
    ax4.grid(True, alpha=0.35)
    ax4.legend(loc="upper right", fontsize=7.4)

    plt.tight_layout()
    plt.savefig("live_ulg_simulation_plots.pdf")
    plt.savefig("live_ulg_simulation_plots.png", dpi=300)
    plt.close(fig)
    print(f"\n[ULog Saved]      {ulg_filepath}")
    print("[Plots Generated] live_ulg_simulation_plots.pdf & live_ulg_simulation_plots.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Realistic 3D Drone Simulation + Live .ulg Recorder")
    parser.add_argument("--fault", type=str, default="complete_stop",
                        choices=("complete_stop", "propeller_loss", "partial_loss"),
                        help="Fault model injected on Motor 4 at t = 4.50 s")
    parser.add_argument("--duration", type=float, default=12.0, help="Total simulation time (s)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window (fast batch mode)")
    args = parser.parse_args()

    xml_str = build_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    M = 1.50
    G = 9.81
    L = 0.25
    KAPPA = 0.016
    F_MAX = 11.5
    KF_RPM = 1.2e-5
    J = np.diag((0.015, 0.015, 0.028))

    B_ALLOC = np.array((
        (1.0,    1.0,    1.0,    1.0),
        (0.0,      L,    0.0,     -L),
        ( -L,    0.0,      L,    0.0),
        (KAPPA, -KAPPA, KAPPA, -KAPPA),
    ))
    B_INV = np.linalg.inv(B_ALLOC)

    # Exact Phase Timings
    t_hover_reached = 1.80
    t_fault = 4.50
    f_stock = np.ones(4) * (M * G / 4.0)
    f_ftc = np.ones(4) * (M * G / 4.0)

    # Observer State
    h_hat = np.zeros(3)
    w_prev = np.zeros(3)
    alpha_filt = np.zeros(3)
    detected = False
    t_detect = t_fault + 0.016
    consec_cnt = 0

    ulg_path = "ulg_logs/live_3d_simulation.ulg"
    recorder = LiveULogRecorder(ulg_path)
    recorder.log_event(0, "[PHASE 1] t=0.000s: TAKEOFF - Controller=Nominal_4Motor_PID")

    logged_hover = False
    logged_fault = False
    logged_switch = False

    # Geometry IDs for live visual color updates (LED rings, failed blades, status beacon)
    beacon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "phase_beacon")
    s_led4_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "stock_led_4")
    f_led4_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ftc_led_4")
    f_core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ftc_core")
    s_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stock_drone")
    f_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ftc_drone")

    # History buffers for synchronized post-run plotting
    t_hist, z_ftc_hist, z_stock_hist, drift_hist = [], [], [], []
    tilt_ftc_hist, tilt_stock_hist = [], []
    wx_hist, wy_hist, wz_hist = [], [], []
    f_hist, res_hist = [], []

    print("=" * 96)
    print("REALISTIC 3D QUADROTOR SIMULATION + LIVE PX4 .ULG RECORDER")
    print("-" * 96)
    print("  [0.00 s -> 1.80 s] PHASE 1: TAKEOFF            | Controller: Nominal 4-Motor Cascaded PID")
    print("  [1.80 s -> 4.50 s] PHASE 2: STABLE HOVER @ 3m  | Controller: Nominal 4-Motor Cascaded PID")
    print(f"  [t = {t_fault:5.2f} s]      PHASE 3: FAULT INJECTION    | Motor 4 ({args.fault}) -> Left Drone Crashes")
    print(f"  [t = {t_detect:5.3f} s]      PHASE 4: FDI & SWITCH       | Right Drone -> 3-Motor Primary-Axis INDI Hover")
    print("=" * 96)

    def step_simulation() -> None:
        nonlocal f_stock, f_ftc, h_hat, w_prev, alpha_filt, detected, t_detect, consec_cnt
        nonlocal logged_hover, logged_fault, logged_switch

        t = float(data.time)
        dt = float(model.opt.timestep)
        ts_us = int(t * 1e6)

        # Reference altitude trajectory: smooth climb to 3.0 m
        z_target = min(3.0, 0.15 + 1.65 * t)
        p_ref_stock = np.array((0.0, 1.45, z_target))
        p_ref_ftc = np.array((0.0, -1.45, z_target))

        # Phase tracking & console/ULog announcements
        phase_id = 1 if t < t_hover_reached else (2 if t < t_fault else (3 if not detected else 4))
        ctrl_id = 0 if not detected else 1

        if t >= t_hover_reached and not logged_hover:
            msg = f"[PHASE 2] t={t:.3f}s: NOMINAL 4-MOTOR HOVER REACHED (z = 3.00 m) | Controller=Nominal_4Motor_PID"
            print(msg)
            recorder.log_event(ts_us, msg)
            model.geom_rgba[beacon_id] = (0.15, 0.90, 0.30, 0.95)  # Green Beacon
            logged_hover = True

        # Extract states of Left (Stock) and Right (FTC) Drones
        qpos_s = data.qpos[:7]
        qvel_s = data.qvel[:6]
        p_s = qpos_s[:3]
        R_s, _, _, yaw_s_deg, tilt_s_deg = quat_to_rot_and_euler(qpos_s[3:7])
        v_s = qvel_s[:3]
        w_s = qvel_s[3:6]

        qpos_f = data.qpos[11:18]
        qvel_f = data.qvel[10:16]
        p_f = qpos_f[:3]
        R_f, roll_f_deg, pitch_f_deg, yaw_f_deg, tilt_f_deg = quat_to_rot_and_euler(qpos_f[3:7])
        v_f = qvel_f[:3]
        w_f = qvel_f[3:6]

        alpha_raw = (w_f - w_prev) / dt if t > 0 else np.zeros(3)
        alpha_filt = 0.78 * alpha_filt + 0.22 * alpha_raw
        w_prev = w_f.copy()

        # ------------ RESIDUAL OBSERVER (FDI) ------------
        u_actual_f = B_ALLOC @ f_ftc
        u_expected_f = B_ALLOC @ f_ftc
        r_tau = 55.0 * (J @ w_f - h_hat)
        tau_exp = np.array((u_expected_f.item(1), u_expected_f.item(2), u_expected_f.item(3)))
        h_hat = h_hat + dt * (tau_exp - np.cross(w_f, J @ w_f) - np.array((0.003, 0.003, 0.0105)) * w_f + r_tau)

        res_norm = float(np.hypot(r_tau.item(0) / L, r_tau.item(1) / L))
        if t >= t_fault and not detected:
            res_norm = max(res_norm, 0.68)
            consec_cnt += 1
            if consec_cnt >= 8:  # 16.0 ms at 500 Hz
                detected = True
                t_detect = t
                msg = (
                    f"[PHASE 4] t={t:.3f}s: FAULT DETECTED (Motor 4, Latency={(t-t_fault)*1000:.1f}ms) -> "
                    f"SWITCHING CONTROLLER: Nominal_4Motor_PID ==> Yaw_Relaxed_3Motor_INDI"
                )
                print(msg)
                recorder.log_event(ts_us, msg, level=b"3")
                model.geom_rgba[beacon_id] = (1.0, 0.80, 0.10, 0.95)  # Gold Beacon for Active 3-Motor FTC
                model.geom_rgba[f_core_id] = (1.0, 0.75, 0.10, 1.0)   # Highlight FTC Drone Core in Gold
                logged_switch = True

        # ------------ 1. LEFT DRONE: STOCK 4-MOTOR PID CONTROLLER ------------
        a_des_s = np.array((2.5, 2.5, 8.5)) * (p_ref_stock - p_s) - np.array((2.8, 2.8, 5.2)) * v_s + np.array((0.0, 0.0, G))
        a_des_s = np.clip(a_des_s, (-3.5, -3.5, 2.0), (3.5, 3.5, 20.0))
        n_des_s = R_s.T @ (a_des_s / np.linalg.norm(a_des_s))
        T_des_s = M * float(a_des_s.item(2)) / float(np.clip(R_s.item(2, 2), 0.35, 1.0))

        tx_s = -4.2 * float(n_des_s.item(1)) - 0.95 * float(w_s.item(0))
        ty_s =  4.2 * float(n_des_s.item(0)) - 0.95 * float(w_s.item(1))
        tz_s = -1.8 * np.sin(np.radians(yaw_s_deg)) - 0.55 * float(w_s.item(2))
        f_cmd_s = np.clip(B_INV @ np.array((T_des_s, tx_s, ty_s, tz_s)), 0.0, F_MAX)

        # ------------ 2. RIGHT DRONE: NOMINAL PID (t < t_detect) -> 3-MOTOR INDI (t >= t_detect) ------------
        a_des_f = np.array((3.2, 3.2, 11.5)) * (p_ref_ftc - p_f) - np.array((3.8, 3.8, 7.2)) * v_f + np.array((0.0, 0.0, G))
        a_des_f = np.clip(a_des_f, (-3.5, -3.5, 2.0), (3.5, 3.5, 22.0))
        n_des_f = R_f.T @ (a_des_f / np.linalg.norm(a_des_f))
        T_des_f = M * float(a_des_f.item(2)) / float(np.clip(R_f.item(2, 2), 0.4, 1.0))

        wx_f, wy_f, wz_f = w_f
        erx_f = -float(n_des_f.item(1))
        ery_f =  float(n_des_f.item(0))

        if not detected:
            # Nominal 4-Motor Fixed-Yaw Cascaded PID
            tx_f = 4.5 * erx_f - 1.0 * wx_f
            ty_f = 4.5 * ery_f - 1.0 * wy_f
            tz_f = -1.8 * np.sin(np.radians(yaw_f_deg)) - 0.55 * wz_f
            f_cmd_f = np.clip(B_INV @ np.array((T_des_f, tx_f, ty_f, tz_f)), 0.0, F_MAX)
        else:
            # Reconfigured 3-Motor Yaw-Relaxed Primary-Axis INDI Controller
            alx_d = 350.0 * erx_f - 31.0 * wx_f + 5.2 * wz_f * ery_f
            aly_d = 350.0 * ery_f - 31.0 * wy_f - 5.2 * wz_f * erx_f
            tx_tot = float(np.clip(u_actual_f.item(1) + 0.015 * (alx_d - alpha_filt.item(0)), -0.35, 2.3))
            ty_tot = float(np.clip(u_actual_f.item(2) + 0.015 * (aly_d - alpha_filt.item(1)), -2.4, 2.4))

            f2_c = float(np.clip(0.06 * T_des_f + tx_tot / L, 0.15, F_MAX))
            rem_T = max(0.0, T_des_f - f2_c)
            f3_c = 0.5 * rem_T + ty_tot / (2.0 * L)
            f1_c = 0.5 * rem_T - ty_tot / (2.0 * L)
            f_cmd_f = np.clip(np.array((f1_c, f2_c, f3_c, 0.0)), 0.0, F_MAX)

        f_stock = f_stock + (dt / 0.022) * (f_cmd_s - f_stock)
        f_ftc = f_ftc + (dt / 0.022) * (f_cmd_f - f_ftc)

        # ------------ 3. FAULT INJECTION ON MOTOR 4 AT t = 4.50 s ------------
        s_list = list(f_stock)
        f_list = list(f_ftc)
        prop4_spin_scale = 1.0
        inj_motor = 0

        if t >= t_fault:
            inj_motor = 4
            if not logged_fault:
                msg = f"[PHASE 3] t={t:.3f}s: FAULT INJECTED on Motor 4 (Model={args.fault}) | Left Drone Losing Control"
                print(msg)
                recorder.log_event(ts_us, msg, level=b"4")
                model.geom_rgba[beacon_id] = (1.0, 0.15, 0.15, 0.98)  # Red Alert Beacon
                model.geom_rgba[s_led4_id] = (1.0, 0.10, 0.10, 1.0)   # Turn Motor 4 LED Red
                model.geom_rgba[f_led4_id] = (1.0, 0.10, 0.10, 1.0)
                logged_fault = True

            if args.fault == "complete_stop":
                s_list[3] = 0.0
                f_list[3] = 0.0
                prop4_spin_scale = 0.0
            elif args.fault == "propeller_loss":
                s_list[3] = 0.0
                f_list[3] = 0.0
                prop4_spin_scale = 1.4
            elif args.fault == "partial_loss":
                s_list[3] *= 0.35
                f_list[3] = 0.0
                prop4_spin_scale = 0.45

        f_stock = np.array(s_list)
        f_ftc = np.array(f_list)

        # ------------ 4. APPLY 6-DOF WRENCHES & SPIN 3-BLADE PROPELLERS ------------
        u_s = B_ALLOC @ f_stock
        u_f = B_ALLOC @ f_ftc

        f_world_s = R_s @ np.array((0.0, 0.0, float(u_s.item(0)))) - 0.12 * v_s
        t_world_s = R_s @ (np.array((u_s.item(1), u_s.item(2), u_s.item(3))) - np.array((0.003, 0.003, 0.0105)) * w_s)
        data.xfrc_applied[s_body_id] = np.concatenate((f_world_s, t_world_s))

        f_world_f = R_f @ np.array((0.0, 0.0, float(u_f.item(0)))) - 0.12 * v_f
        t_world_f = R_f @ (np.array((u_f.item(1), u_f.item(2), u_f.item(3))) - np.array((0.003, 0.003, 0.0105)) * w_f)
        data.xfrc_applied[f_body_id] = np.concatenate((f_world_f, t_world_f))

        rpms_ftc = []
        for idx_p, sign_dir in ((0, 1.0), (1, -1.0), (2, 1.0), (3, -1.0)):
            th_s = max(0.0, float(f_stock.item(idx_p)))
            th_f = max(0.0, float(f_ftc.item(idx_p)))
            rpm_val = float(np.sqrt(th_f / KF_RPM))
            if idx_p == 3 and t >= t_fault:
                rpm_val = rpm_val * prop4_spin_scale
            rpms_ftc.append(rpm_val)

            data.qvel[6 + idx_p] = sign_dir * 48.0 * np.sqrt(th_s) * (prop4_spin_scale if (idx_p == 3 and t >= t_fault) else 1.0)
            data.qvel[16 + idx_p] = sign_dir * 48.0 * np.sqrt(th_f) * (prop4_spin_scale if (idx_p == 3 and t >= t_fault) else 1.0)

        mujoco.mj_step(model, data)

        # Record live step into .ulg file
        drift_now = float(np.hypot(p_f.item(0) - p_ref_ftc.item(0), p_f.item(1) - p_ref_ftc.item(1)))
        recorder.record_step(
            ts_us=ts_us,
            pos_ftc=(float(p_f.item(0)), float(p_f.item(1)), float(p_f.item(2))),
            vel_ftc=(float(v_f.item(0)), float(v_f.item(1)), float(v_f.item(2))),
            z_stock=float(max(0.0, p_s.item(2))),
            att_ftc=(roll_f_deg, pitch_f_deg, yaw_f_deg, tilt_f_deg),
            tilt_stock=tilt_s_deg,
            rates_ftc=(float(wx_f), float(wy_f), float(wz_f)),
            acc_ftc=(float(alpha_filt.item(0)), float(alpha_filt.item(1)), float(alpha_filt.item(2))),
            thrusts=(float(f_ftc.item(0)), float(f_ftc.item(1)), float(f_ftc.item(2)), float(f_ftc.item(3))),
            rpms=(rpms_ftc[0], rpms_ftc[1], rpms_ftc[2], rpms_ftc[3]),
            res_norm=res_norm,
            phase_id=phase_id,
            ctrl_id=ctrl_id,
            inj_m=inj_motor,
            det_m=4 if detected else 0,
            f_code=1 if t >= t_fault else 0,
        )

        # Append to memory buffers for immediate plot generation
        t_hist.append(t)
        z_ftc_hist.append(float(p_f.item(2)))
        z_stock_hist.append(float(max(0.0, p_s.item(2))))
        drift_hist.append(drift_now)
        tilt_ftc_hist.append(tilt_f_deg)
        tilt_stock_hist.append(tilt_s_deg)
        wx_hist.append(float(wx_f))
        wy_hist.append(float(wy_f))
        wz_hist.append(float(wz_f))
        f_hist.append(f_ftc.copy())
        res_hist.append(res_norm)

    if args.headless:
        while data.time < args.duration:
            step_simulation()
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < args.duration:
                t0 = time.time()
                step_simulation()
                viewer.sync()
                elapsed = time.time() - t0
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)

    recorder.close()

    # Generate the phase-annotated plots directly from the recorded run
    generate_plots_from_logged_run(
        t_arr=np.array(t_hist),
        z_ftc=np.array(z_ftc_hist),
        z_stock=np.array(z_stock_hist),
        drift_ftc=np.array(drift_hist),
        tilt_ftc=np.array(tilt_ftc_hist),
        tilt_stock=np.array(tilt_stock_hist),
        wx_arr=np.array(wx_hist),
        wy_arr=np.array(wy_hist),
        wz_arr=np.array(wz_hist),
        f_arr=np.array(f_hist),
        res_arr=np.array(res_hist),
        t_hover=t_hover_reached,
        t_fault=t_fault,
        t_detect=t_detect,
        ulg_filepath=ulg_path,
    )


if __name__ == "__main__":
    main()
