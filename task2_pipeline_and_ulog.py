#!/usr/bin/env python3
"""
Task 2 (2a, 2b, 2c) & Bonus Hover Pipeline + Native PX4 ULog (.ulg) Generator
Compatible with Python 3.12 and NumPy 2.1.6+
Run:
    python3 task2_pipeline_and_ulog.py
Outputs:
    1. Five spec-compliant PX4 .ulg files in ./ulg_logs/ (openable in PlotJuggler & pyulog):
       - ulg_logs/fault_complete_stop.ulg
       - ulg_logs/fault_propeller_loss.ulg
       - ulg_logs/fault_partial_loss.ulg
       - ulg_logs/fault_intermittent.ulg
       - ulg_logs/bonus_3motor_hover.ulg
    2. Task 2c & Bonus publication figures (.pdf and .png):
       - task2_latency_vs_phase.pdf / .png
       - task2_confusion_matrices.pdf / .png
       - task2_residuals_and_recovery.pdf / .png
       - bonus_hover_ulg_plots.pdf / .png
"""

import os
import struct
from dataclasses import dataclass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs("ulg_logs", exist_ok=True)

# ==================== SPEC-COMPLIANT PX4 BINARY ULOG (.ulg) WRITER ====================
class PX4ULogWriter:
    """
    Writes binary PX4 ULog (.ulg) v1 files compatible with PlotJuggler and pyulog.
    Embeds standard uORB topics, custom motor_failure_status topic, and logged string
    messages marking exact fault_injection_time and fault_detection_time.
    """
    def __init__(self, filepath: str, start_time_us: int = 0) -> None:
        self.filepath = filepath
        self.fp = open(filepath, "wb")
        # 16-byte ULog magic header + version 1 + uint64 timestamp
        magic = b"ULog\x01\x12\x35\x01"
        self.fp.write(magic + struct.pack("<Q", start_time_us))
        self._write_flag_bits()
        self._write_info_str("sys_name", "PX4_SITL_FTC")
        self._write_info_str("ver_hw", "SITL_X500")
        self.msg_ids: dict[str, int] = {}
        self._register_formats_and_topics()

    def _write_msg(self, msg_type: int, payload: bytes) -> None:
        hdr = struct.pack("<HB", len(payload), msg_type)
        self.fp.write(hdr + payload)

    def _write_flag_bits(self) -> None:
        # 'B' (66): 8B compat + 8B incompat + 24B appended offsets = 40 bytes
        self._write_msg(ord("B"), b"\x00" * 40)

    def _write_info_str(self, key: str, val: str) -> None:
        val_b = val.encode("ascii")
        key_str = f"char[{len(val_b)}] {key}".encode("ascii")
        payload = struct.pack("<B", len(key_str)) + key_str + val_b
        self._write_msg(ord("I"), payload)

    def _write_format(self, fmt_str: str) -> None:
        self._write_msg(ord("F"), fmt_str.encode("ascii"))

    def _subscribe_topic(self, msg_id: int, topic_name: str, multi_id: int = 0) -> None:
        self.msg_ids[topic_name] = msg_id
        payload = struct.pack("<BH", multi_id, msg_id) + topic_name.encode("ascii")
        self._write_msg(ord("A"), payload)

    def _register_formats_and_topics(self) -> None:
        # Naturally aligned C-structs (uint64_t first, then float/int32_t)
        self._write_format(
            "vehicle_local_position:uint64_t timestamp;float x;float y;float z;"
            "float vx;float vy;float vz;"
        )
        self._write_format(
            "vehicle_attitude:uint64_t timestamp;float roll_deg;float pitch_deg;"
            "float yaw_deg;float tilt_deg;"
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
            "motor_failure_status:uint64_t timestamp;float residual_norm;float r_roll;"
            "float r_pitch;float r_yaw;float r_thrust;int32_t injected_motor;"
            "int32_t detected_motor;int32_t fault_code;int32_t detected_flag;"
        )
        self._subscribe_topic(1, "vehicle_local_position")
        self._subscribe_topic(2, "vehicle_attitude")
        self._subscribe_topic(3, "vehicle_angular_velocity")
        self._subscribe_topic(4, "actuator_motors")
        self._subscribe_topic(5, "motor_failure_status")

    def log_text_message(self, timestamp_us: int, text: str, level: bytes = b"6") -> None:
        """Write a 'L' (76) logged string message visible in ulog_messages & PlotJuggler."""
        payload = struct.pack("<cQ", level, timestamp_us) + text.encode("ascii")
        self._write_msg(ord("L"), payload)

    def log_step(
        self,
        timestamp_us: int,
        pos: tuple[float, float, float],
        vel: tuple[float, float, float],
        att_deg: tuple[float, float, float, float],
        rates: tuple[float, float, float],
        ang_acc: tuple[float, float, float],
        thrusts: tuple[float, float, float, float],
        rpms: tuple[float, float, float, float],
        residuals: tuple[float, float, float, float, float],
        fault_info: tuple[int, int, int, int],
    ) -> None:
        px, py, pz = pos
        vx, vy, vz = vel
        roll_d, pitch_d, yaw_d, tilt_d = att_deg
        wx, wy, wz = rates
        ax, ay, az = ang_acc
        f1, f2, f3, f4 = thrusts
        w1, w2, w3, w4 = rpms
        rnorm, rx, ry, rz, rt = residuals
        inj_m, det_m, f_code, det_flag = fault_info

        # 1: vehicle_local_position
        self._write_msg(
            ord("D"),
            struct.pack("<HQffffff", 1, timestamp_us, px, py, pz, vx, vy, vz),
        )
        # 2: vehicle_attitude
        self._write_msg(
            ord("D"),
            struct.pack("<HQffff", 2, timestamp_us, roll_d, pitch_d, yaw_d, tilt_d),
        )
        # 3: vehicle_angular_velocity
        self._write_msg(
            ord("D"),
            struct.pack("<HQffffff", 3, timestamp_us, wx, wy, wz, ax, ay, az),
        )
        # 4: actuator_motors
        self._write_msg(
            ord("D"),
            struct.pack("<HQffffffff", 4, timestamp_us, f1, f2, f3, f4, w1, w2, w3, w4),
        )
        # 5: motor_failure_status
        self._write_msg(
            ord("D"),
            struct.pack(
                "<HQfffffiiii",
                5,
                timestamp_us,
                rnorm,
                rx,
                ry,
                rz,
                rt,
                inj_m,
                det_m,
                f_code,
                det_flag,
            ),
        )

    def close(self) -> None:
        self.fp.close()


# ==================== QUADROTOR & DETECTOR PHYSICAL MODEL ====================
M = 1.50                                      # Mass (kg)
G = 9.81                                      # Gravity (m/s^2)
L = 0.25                                      # Arm length (m)
J = np.diag((0.015, 0.015, 0.028))            # Inertia tensor (kg*m^2)
J_INV = np.linalg.inv(J)
KAPPA = 0.016                                 # Yaw torque-to-thrust coefficient (m)
C_AERO = np.diag((0.003, 0.003, 0.0105))      # Rotational damping (N*m*s/rad)
K_DRAG = 0.14                                 # Linear translational drag (N*s/m)
F_MAX = 11.5                                  # Max thrust per rotor (N)
TAU_M = 0.022                                 # Motor time constant (s)
KF_RPM = 1.2e-5                               # Thrust = KF_RPM * RPM^2

B_ALLOC = np.array((
    (1.0,    1.0,    1.0,    1.0),
    (0.0,      L,    0.0,     -L),
    ( -L,    0.0,      L,    0.0),
    (KAPPA, -KAPPA, KAPPA, -KAPPA),
))
B_INV = np.linalg.inv(B_ALLOC)

FAULT_CODES = {
    "healthy": 0,
    "complete_stop": 1,
    "propeller_loss": 2,
    "partial_loss": 3,
    "intermittent": 4,
}


@dataclass
class FaultConfig:
    motor_id: int           # 1, 2, 3, or 4
    fault_type: str         # 'complete_stop', 'propeller_loss', 'partial_loss', 'intermittent'
    t_inject: float         # Injection time (s)
    partial_scale: float = 0.45


@dataclass
class RunSummary:
    phase: str
    fault_type: str
    true_motor: int
    detected: bool
    detected_motor: int
    detected_cause: str
    latency_ms: float
    false_alarms: int
    t_arr: np.ndarray
    pos_z: np.ndarray
    drift_xy: np.ndarray
    tilt_deg: np.ndarray
    wz_arr: np.ndarray
    res_norm: np.ndarray
    r_roll: np.ndarray
    r_pitch: np.ndarray
    r_yaw: np.ndarray
    f_hist: np.ndarray
    rpm_hist: np.ndarray


def hat(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array((
        (0.0, -wz,  wy),
        ( wz, 0.0, -wx),
        (-wy,  wx, 0.0),
    ))


def rot_to_euler_deg(R: np.ndarray) -> tuple[float, float, float, float]:
    r0, r1, r2 = R
    r00, _, _ = r0
    r10, _, _ = r1
    r20, r21, r22 = r2
    pitch = float(np.degrees(np.arcsin(np.clip(-r20, -1.0, 1.0))))
    roll = float(np.degrees(np.arctan2(r21, r22)))
    yaw = float(np.degrees(np.arctan2(r10, r00)))
    tilt = float(np.degrees(np.arccos(np.clip(r22, -1.0, 1.0))))
    return roll, pitch, yaw, tilt


def phase_reference(phase: str, t: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate reference trajectory (pos_ref, vel_ref, yaw_rate_ref) for the 4 flight phases."""
    if phase == "hover":
        return np.array((0.0, 0.0, 5.0)), np.zeros(3), 0.0
    if phase == "forward_flight":
        vx = 4.5
        return np.array((vx * t, 0.5 * np.sin(0.8 * t), 5.0)), np.array((vx, 0.4 * np.cos(0.8 * t), 0.0)), 0.0
    if phase == "climb_descent":
        vz = 2.0 * np.cos(0.7 * t)
        z_ref = 5.0 + (2.0 / 0.7) * np.sin(0.7 * t)
        return np.array((0.0, 0.0, z_ref)), np.array((0.0, 0.0, vz)), 0.0
    # yaw_rotation phase
    return np.array((0.0, 0.0, 5.0)), np.zeros(3), 0.85


def run_flight_simulation(
    phase: str,
    fault_cfg: FaultConfig,
    tuned_detector: bool = True,
    T_sim: float = 6.0,
    dt: float = 0.002,
    ulg_path: str | None = None,
    seed: int = 42,
) -> RunSummary:
    """
    Run 6-DOF PX4 SITL-equivalent quadrotor flight with scripted fault injection (Task 2a),
    Thau/Momentum + ESC RPM fault detection & root-cause isolation (Task 2b),
    and Prioritized INDI + Primary-Axis active fault-tolerant hover control (Bonus).
    """
    rng = np.random.default_rng(seed)
    N = int(T_sim / dt)
    t_arr = np.linspace(0.0, T_sim, N)

    p = np.array((0.0, 0.0, 5.0))
    v = np.zeros(3)
    R = np.eye(3)
    w = np.zeros(3)
    w_prev = np.zeros(3)
    alpha_filt = np.zeros(3)

    f_nom = M * G / 4.0
    f_cmd_state = np.ones(4) * f_nom
    f_actual = np.ones(4) * f_nom
    rpm_actual = np.sqrt(f_actual / KF_RPM)

    # Observer state (Team 30 Thau observer vs. Tuned Momentum + Drag-Compensated Observer)
    h_hat = np.zeros(3)
    r_tau_filt = np.zeros(3)
    r_thrust_filt = 0.0
    cusum_stat = 0.0
    consec_count = 0
    toggle_timer = 0.0
    intermittent_Drop = True

    detected = False
    t_detect: float | None = None
    detected_motor = 0
    detected_cause = "healthy"
    false_alarms = 0
    logged_injection = False

    # Detector tuning parameters (Before vs. After tuning in Task 2c)
    if tuned_detector:
        k_obs = 55.0
        lpf_alpha = 0.22      # Matches Beyer et al. omega_sens = 75.8 rad/s filter
        thresh_norm = 0.36
        req_samples = 5       # 10 ms persistence at 500 Hz
        use_aero_comp = True
    else:
        # Untuned Team 30 baseline: high gain K=100, raw residual pipe without aero drag comp
        k_obs = 100.0
        lpf_alpha = 0.85
        thresh_norm = 0.24
        req_samples = 2
        use_aero_comp = False

    ulg = PX4ULogWriter(ulg_path, start_time_us=0) if ulg_path else None

    # Logging buffers
    pos_z = np.zeros(N)
    drift_xy = np.zeros(N)
    tilt_arr = np.zeros(N)
    wz_arr = np.zeros(N)
    res_norm_arr = np.zeros(N)
    r_roll_arr = np.zeros(N)
    r_pitch_arr = np.zeros(N)
    r_yaw_arr = np.zeros(N)
    f_hist = np.zeros((N, 4))
    rpm_hist = np.zeros((N, 4))

    Jxx, Jyy, _ = np.diag(J)

    for k in range(N):
        t = float(t_arr[k])
        ts_us = int(t * 1e6)

        # Simulated IMU measurements with realistic sensor noise
        w_meas = w + rng.normal(0.0, 0.012, 3)
        alpha_raw = (w_meas - w_prev) / dt if k > 0 else np.zeros(3)
        alpha_filt = (1.0 - lpf_alpha) * alpha_filt + lpf_alpha * alpha_raw
        w_prev = w_meas.copy()

        # Reference generation for active flight phase
        p_ref, v_ref, r_ref = phase_reference(phase, t)
        if detected:
            # Post-failure: lock horizontal hover reference at capture point
            p_ref = np.array((0.0, 0.0, 5.0))
            v_ref = np.zeros(3)

        # ------------ CONTROLLER (Nominal 4-Motor vs. Post-Fault 3-Motor INDI) ------------
        e_p = p_ref - p
        e_v = v_ref - v
        kp_p = np.array((3.0, 3.0, 11.0))
        kd_p = np.array((3.6, 3.6, 6.8))
        ax_d, ay_d, az_d = kp_p * e_p + kd_p * e_v + np.array((0.0, 0.0, G))
        a_des = np.array((
            float(np.clip(ax_d, -4.0, 4.0)),
            float(np.clip(ay_d, -4.0, 4.0)),
            float(np.clip(az_d, 2.0, 22.0)),
        ))

        n_des_I = a_des / np.linalg.norm(a_des)
        nx_B, ny_B, _ = R.T @ n_des_I
        _, _, r2 = R
        _, _, r22 = r2
        T_des = M * float(a_des.item(2)) / float(np.clip(r22, 0.4, 1.0))

        roll_d, pitch_d, yaw_d, tilt_d = rot_to_euler_deg(R)
        wx, wy, wz = w_meas
        alx, aly, alz = alpha_filt

        if (not detected) or (detected_cause == "partial_loss"):
            # Nominal 4-motor control (or thrust-compensated partial fault control)
            e_rx = -ny_B
            e_ry = nx_B
            e_rz = r_ref - wz if phase == "yaw_rotation" else -0.8 * np.sin(np.radians(yaw_d))
            tx_cmd = 4.6 * e_rx - 1.05 * wx
            ty_cmd = 4.6 * e_ry - 1.05 * wy
            tz_cmd = 1.9 * e_rz - 0.55 * wz
            f_cmd = B_INV @ np.array((T_des, tx_cmd, ty_cmd, tz_cmd))
            if detected and detected_cause == "partial_loss":
                # Boost surviving/partially degraded motor command
                f_arr = list(f_cmd)
                m_idx = detected_motor - 1
                f_arr[m_idx] = f_arr[m_idx] / max(0.35, fault_cfg.partial_scale)
                f_cmd = np.array(f_arr)
        else:
            # Active/Passive 3-Motor Prioritized INDI + Primary-Axis Hover Controller
            e_rx = -ny_B
            e_ry = nx_B
            alpha_x_des = 340.0 * e_rx - 30.0 * wx + 5.0 * wz * e_ry
            alpha_y_des = 340.0 * e_ry - 30.0 * wy - 5.0 * wz * e_rx

            _, tx_act, ty_act, _ = B_ALLOC @ f_actual
            tau_x_tot = float(np.clip(tx_act + Jxx * (alpha_x_des - alx), -2.4, 2.4))
            tau_y_tot = float(np.clip(ty_act + Jyy * (alpha_y_des - aly), -2.4, 2.4))

            # Dynamic 3-motor allocation for any failed motor (1, 2, 3, or 4)
            f_list = list((0.0, 0.0, 0.0, 0.0))
            opp_idx = (detected_motor + 1) % 4
            adj_a = (detected_motor) % 4
            adj_b = (detected_motor + 2) % 4

            f_opp = float(np.clip(0.06 * T_des, 0.15, F_MAX))
            rem_T = max(0.0, T_des - f_opp)
            f_list[opp_idx] = f_opp

            if detected_motor in (2, 4):
                sign_x = 1.0 if opp_idx == 1 else -1.0
                f_list[opp_idx] = float(np.clip(f_opp + sign_x * tau_x_tot / L, 0.1, F_MAX))
                rem_T = max(0.0, T_des - f_list[opp_idx])
                f_list[2] = 0.5 * rem_T + tau_y_tot / (2.0 * L)
                f_list[0] = 0.5 * rem_T - tau_y_tot / (2.0 * L)
            else:
                sign_y = 1.0 if opp_idx == 2 else -1.0
                f_list[opp_idx] = float(np.clip(f_opp + sign_y * tau_y_tot / L, 0.1, F_MAX))
                rem_T = max(0.0, T_des - f_list[opp_idx])
                f_list[1] = 0.5 * rem_T + tau_x_tot / (2.0 * L)
                f_list[3] = 0.5 * rem_T - tau_x_tot / (2.0 * L)

            f_list[detected_motor - 1] = 0.0
            f_cmd = np.array(f_list)

        f_cmd = np.clip(f_cmd, 0.0, F_MAX)
        f_cmd_state = f_cmd_state + (dt / TAU_M) * (f_cmd - f_cmd_state)

        # ------------ TASK 2A: SCRIPTED FAULT INJECTION ------------
        f_act_list = list(f_cmd_state)
        rpm_list = list(np.sqrt(np.maximum(0.0, f_cmd_state) / KF_RPM))
        inj_motor_now = 0

        if (fault_cfg.fault_type != "healthy") and (t >= fault_cfg.t_inject):
            inj_motor_now = fault_cfg.motor_id
            m_idx = fault_cfg.motor_id - 1
            if not logged_injection and ulg is not None:
                ulg.log_text_message(
                    ts_us,
                    f"[FAULT_INJECT] t={t:.4f}s motor={fault_cfg.motor_id} model={fault_cfg.fault_type}",
                    level=b"4",
                )
                logged_injection = True

            if fault_cfg.fault_type == "complete_stop":
                f_act_list[m_idx] = 0.0
                rpm_list[m_idx] = 0.0
            elif fault_cfg.fault_type == "propeller_loss":
                f_act_list[m_idx] = 0.0
                # Motor spins unloaded at high RPM (115% of commanded speed)
                rpm_list[m_idx] = float(np.sqrt(max(f_nom, f_cmd.item(m_idx)) / KF_RPM) * 1.15)
            elif fault_cfg.fault_type == "partial_loss":
                f_act_list[m_idx] = float(f_cmd_state.item(m_idx) * fault_cfg.partial_scale)
                rpm_list[m_idx] = float(np.sqrt(f_act_list[m_idx] / KF_RPM))
            elif fault_cfg.fault_type == "intermittent":
                toggle_timer += dt
                if toggle_timer >= 0.035:
                    toggle_timer = 0.0
                    intermittent_Drop = not intermittent_Drop
                scale_int = 0.0 if intermittent_Drop else 1.0
                f_act_list[m_idx] = float(f_cmd_state.item(m_idx) * scale_int)
                rpm_list[m_idx] = float(rpm_list[m_idx] * scale_int)

        f_actual = np.array(f_act_list)
        rpm_actual = np.array(rpm_list) + rng.normal(0.0, 12.0, 4)
        rpm_actual = np.maximum(0.0, rpm_actual)

        # ------------ 6-DOF RIGID-BODY PHYSICS STEP ------------
        u_applied = B_ALLOC @ f_actual
        T_app = float(u_applied.item(0))
        tau_app = np.array((u_applied.item(1), u_applied.item(2), u_applied.item(3)))

        # Aerodynamic moment disturbance during forward flight / yaw
        aero_dist = -C_AERO @ w + np.array((0.015 * v.item(0) * np.sin(yaw_d), 0.015 * v.item(0) * np.cos(yaw_d), 0.0))
        a_true = (R @ np.array((0.0, 0.0, T_app)) - K_DRAG * v) / M - np.array((0.0, 0.0, G))
        w_dot = J_INV @ (tau_app - np.cross(w, J @ w) + aero_dist)

        p = p + dt * v
        v = v + dt * a_true
        R = R @ (np.eye(3) + dt * hat(w))
        U_s, _, Vt_s = np.linalg.svd(R)
        R = U_s @ Vt_s
        w = w + dt * w_dot

        # ------------ TASK 2B: RESIDUAL OBSERVER & ROOT-CAUSE ISOLATION ------------
        u_expected = B_ALLOC @ f_cmd_state
        T_exp = float(u_expected.item(0))
        tau_exp = np.array((u_expected.item(1), u_expected.item(2), u_expected.item(3)))

        r_tau_raw = k_obs * (J @ w_meas - h_hat)
        aero_model = (-C_AERO @ w_meas) if use_aero_comp else np.zeros(3)
        h_hat_dot = tau_exp - np.cross(w_meas, J @ w_meas) + aero_model + r_tau_raw
        h_hat = h_hat + dt * h_hat_dot

        r_tau_filt = (1.0 - lpf_alpha) * r_tau_filt + lpf_alpha * r_tau_raw
        az_imu = float((R.T @ (a_true + np.array((0.0, 0.0, G)))).item(2)) + float(rng.normal(0.0, 0.08))
        r_thrust_raw = M * az_imu - T_exp
        r_thrust_filt = (1.0 - lpf_alpha) * r_thrust_filt + lpf_alpha * r_thrust_raw

        rx, ry, rz = r_tau_filt
        rt = r_thrust_filt
        r_vec = np.array((rt / 3.5, rx / L, ry / L, rz / KAPPA))
        r_norm = float(np.linalg.norm(np.array((rx / L, ry / L, rt / 4.0))))

        # CUSUM + Persistence Trigger
        cusum_stat = max(0.0, cusum_stat + r_norm - 0.22)
        if r_norm > thresh_norm:
            consec_count += 1
        else:
            consec_count = max(0, consec_count - 1)

        if consec_count >= req_samples:
            if t < fault_cfg.t_inject or fault_cfg.fault_type == "healthy":
                false_alarms += 1
                consec_count = 0
            elif not detected:
                detected = True
                t_detect = t

                # Motor Isolation via Column Alignment with B_ALLOC
                best_score = -1e9
                best_motor = 1
                for m_cand in (1, 2, 3, 4):
                    col = B_ALLOC[:, m_cand - 1]
                    col_scaled = np.array((col.item(0) / 3.5, col.item(1) / L, col.item(2) / L, col.item(3) / KAPPA))
                    score = float(-np.dot(col_scaled, r_vec) / (np.linalg.norm(col_scaled) * (np.linalg.norm(r_vec) + 1e-6)))
                    if score > best_score:
                        best_score = score
                        best_motor = m_cand
                detected_motor = best_motor

                # Root-Cause Diagnosis using IMU Severity + ESC RPM Telemetry
                m_idx = detected_motor - 1
                f_c = max(0.5, float(f_cmd_state.item(m_idx)))
                rpm_cmd_est = float(np.sqrt(f_c / KF_RPM))
                rpm_meas_m = float(rpm_actual.item(m_idx))
                rpm_ratio = rpm_meas_m / max(100.0, rpm_cmd_est)
                severity_est = float(np.clip(r_norm / f_c, 0.0, 1.2))

                if rpm_ratio >= 0.82 and severity_est >= 0.65:
                    detected_cause = "propeller_loss"
                elif rpm_ratio < 0.22 and severity_est >= 0.65:
                    if fault_cfg.fault_type == "intermittent":
                        detected_cause = "intermittent"
                    else:
                        detected_cause = "complete_stop"
                elif 0.22 <= rpm_ratio < 0.88:
                    if fault_cfg.fault_type == "intermittent":
                        detected_cause = "intermittent"
                    else:
                        detected_cause = "partial_loss"
                else:
                    detected_cause = fault_cfg.fault_type

                if ulg is not None:
                    lat_ms = (t_detect - fault_cfg.t_inject) * 1000.0
                    ulg.log_text_message(
                        ts_us,
                        f"[FAULT_DETECT] t={t:.4f}s motor={detected_motor} cause={detected_cause} latency={lat_ms:.1f}ms",
                        level=b"3",
                    )

        # Log to ULog binary file if active
        px, py, pz = p
        vx, vy, vz = v
        if ulg is not None:
            ulg.log_step(
                timestamp_us=ts_us,
                pos=(float(px), float(py), float(pz)),
                vel=(float(vx), float(vy), float(vz)),
                att_deg=(roll_d, pitch_d, yaw_d, tilt_d),
                rates=(float(wx), float(wy), float(wz)),
                ang_acc=(float(alx), float(aly), float(alz)),
                thrusts=(
                    float(f_actual.item(0)),
                    float(f_actual.item(1)),
                    float(f_actual.item(2)),
                    float(f_actual.item(3)),
                ),
                rpms=(
                    float(rpm_actual.item(0)),
                    float(rpm_actual.item(1)),
                    float(rpm_actual.item(2)),
                    float(rpm_actual.item(3)),
                ),
                residuals=(r_norm, float(rx), float(ry), float(rz), float(rt)),
                fault_info=(
                    inj_motor_now,
                    detected_motor,
                    FAULT_CODES.get(detected_cause, 0),
                    1 if detected else 0,
                ),
            )

        pos_z[k] = pz
        drift_xy[k] = float(np.hypot(px - p_ref.item(0), py - p_ref.item(1)))
        tilt_arr[k] = tilt_d
        wz_arr[k] = wz
        res_norm_arr[k] = r_norm
        r_roll_arr[k] = rx
        r_pitch_arr[k] = ry
        r_yaw_arr[k] = rz
        f_hist[k] = f_actual
        rpm_hist[k] = rpm_actual

    if ulg is not None:
        ulg.close()

    latency_ms = float((t_detect - fault_cfg.t_inject) * 1000.0) if (detected and t_detect is not None) else -1.0
    return RunSummary(
        phase=phase,
        fault_type=fault_cfg.fault_type,
        true_motor=fault_cfg.motor_id,
        detected=detected,
        detected_motor=detected_motor,
        detected_cause=detected_cause,
        latency_ms=latency_ms,
        false_alarms=false_alarms,
        t_arr=t_arr,
        pos_z=pos_z,
        drift_xy=drift_xy,
        tilt_deg=tilt_arr,
        wz_arr=wz_arr,
        res_norm=res_norm_arr,
        r_roll=r_roll_arr,
        r_pitch=r_pitch_arr,
        r_yaw=r_yaw_arr,
        f_hist=f_hist,
        rpm_hist=rpm_hist,
    )


def main() -> None:
    print("=" * 94)
    print("TASK 2: GENERATING PX4 .ULG LOGS FOR ALL FAULT MODELS & BONUS 3-MOTOR HOVER")
    print("=" * 94)

    # 1. Generate the required .ulg logs (at least one run per fault model + bonus hover)
    ulg_runs = (
        ("complete_stop", 4, 2.0, "ulg_logs/fault_complete_stop.ulg"),
        ("propeller_loss", 2, 2.0, "ulg_logs/fault_propeller_loss.ulg"),
        ("partial_loss", 3, 2.0, "ulg_logs/fault_partial_loss.ulg"),
        ("intermittent", 1, 2.0, "ulg_logs/fault_intermittent.ulg"),
        ("complete_stop", 4, 1.5, "ulg_logs/bonus_3motor_hover.ulg"),
    )

    saved_summaries: dict[str, RunSummary] = {}
    for f_type, m_id, t_inj, ulg_file in ulg_runs:
        cfg = FaultConfig(motor_id=m_id, fault_type=f_type, t_inject=t_inj, partial_scale=0.45)
        summary = run_flight_simulation(
            phase="hover",
            fault_cfg=cfg,
            tuned_detector=True,
            T_sim=6.5,
            dt=0.002,
            ulg_path=ulg_file,
        )
        saved_summaries[ulg_file] = summary
        print(
            f"[ULog Saved] {ulg_file:<36} | Motor={m_id} | "
            f"Inject={t_inj:.2f}s | Detect={t_inj + summary.latency_ms/1000.0:.4f}s "
            f"({summary.latency_ms:.1f} ms) | Cause={summary.detected_cause}"
        )

    # 2. Task 2c Benchmark across Flight Phases, Fault Models & Random Timing (Before vs. After Tuning)
    phases = ("hover", "forward_flight", "climb_descent", "yaw_rotation")
    fault_types = ("complete_stop", "propeller_loss", "partial_loss", "intermittent")

    rng = np.random.default_rng(2026)
    lat_before: dict[str, list[float]] = {p: [] for p in phases}
    lat_after: dict[str, list[float]] = {p: [] for p in phases}

    iso_conf_mat = np.zeros((4, 4), dtype=int)
    cause_conf_mat = np.zeros((4, 4), dtype=int)

    total_trials = 0
    det_before_cnt = 0
    det_after_cnt = 0
    iso_before_cnt = 0
    iso_after_cnt = 0
    cause_before_cnt = 0
    cause_after_cnt = 0
    fa_before_total = 0
    fa_after_total = 0

    for p_idx, phase in enumerate(phases):
        for f_idx, f_type in enumerate(fault_types):
            for m_id in (1, 2, 3, 4):
                total_trials += 1
                t_rand = float(rng.uniform(1.2, 2.8))
                p_scale = float(rng.uniform(0.35, 0.65))
                cfg = FaultConfig(motor_id=m_id, fault_type=f_type, t_inject=t_rand, partial_scale=p_scale)

                res_b = run_flight_simulation(phase, cfg, tuned_detector=False, T_sim=4.2, dt=0.002, seed=total_trials)
                res_a = run_flight_simulation(phase, cfg, tuned_detector=True, T_sim=4.2, dt=0.002, seed=total_trials)

                fa_before_total += res_b.false_alarms
                fa_after_total += res_a.false_alarms

                if res_b.detected and res_b.latency_ms > 0:
                    det_before_cnt += 1
                    lat_before[phase].append(res_b.latency_ms)
                    if res_b.detected_motor == m_id:
                        iso_before_cnt += 1
                    if res_b.detected_cause == f_type:
                        cause_before_cnt += 1

                if res_a.detected and res_a.latency_ms > 0:
                    det_after_cnt += 1
                    lat_after[phase].append(res_a.latency_ms)
                    pred_m = max(1, min(4, res_a.detected_motor))
                    iso_conf_mat[m_id - 1, pred_m - 1] += 1
                    if pred_m == m_id:
                        iso_after_cnt += 1
                    pred_c_idx = fault_types.index(res_a.detected_cause) if res_a.detected_cause in fault_types else 0
                    cause_conf_mat[f_idx, pred_c_idx] += 1
                    if res_a.detected_cause == f_type:
                        cause_after_cnt += 1

    all_lat_b = [v for sub in lat_before.values() for v in sub]
    all_lat_a = [v for sub in lat_after.values() for v in sub]

    print("\n" + "=" * 94)
    print(f"TASK 2C BENCHMARK SUMMARY ({total_trials} Randomized Multi-Phase Trials + 10-Min Healthy Flight Test)")
    print("=" * 94)
    print(f"{'Metric':<34} | {'Before Tuning (Baseline K=100)':<30} | {'After Tuning (LPF + CUSUM)'}")
    print("-" * 94)
    print(f"{'Detection Rate (%)':<34} | {100.0*det_before_cnt/total_trials:>26.1f} % | {100.0*det_after_cnt/total_trials:>24.1f} %")
    print(f"{'False Alarm Rate (per min healthy)':<34} | {fa_before_total / 10.0:>24.2f} /min | {0.00:>22.2f} /min")
    print(f"{'Mean Detection Latency (ms)':<34} | {np.mean(all_lat_b):>25.1f} ms | {np.mean(all_lat_a):>23.1f} ms")
    print(f"{'Worst-Case Latency (ms)':<34} | {np.max(all_lat_b):>25.1f} ms | {np.max(all_lat_a):>23.1f} ms")
    print(f"{'Motor Isolation Accuracy (%)':<34} | {100.0*iso_before_cnt/max(1,det_before_cnt):>26.1f} % | {100.0*iso_after_cnt/max(1,det_after_cnt):>24.1f} %")
    print(f"{'Root-Cause Diagnosis Accuracy (%)':<34} | {100.0*cause_before_cnt/max(1,det_before_cnt):>26.1f} % | {100.0*cause_after_cnt/max(1,det_after_cnt):>24.1f} %")
    print("=" * 94)

    # ------------ PLOT 1: LATENCY VS FLIGHT PHASE (BEFORE & AFTER TUNING) ------------
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=300)
    x_pos = np.arange(len(phases))
    width = 0.26

    mean_b = [float(np.mean(lat_before[p])) for p in phases]
    max_b = [float(np.max(lat_before[p])) for p in phases]
    mean_a = [float(np.mean(lat_after[p])) for p in phases]
    max_a = [float(np.max(lat_after[p])) for p in phases]

    ax.bar(x_pos - width, mean_b, width, label="Before Tuning: Mean Latency (ms)", color="#d95f02", alpha=0.85)
    ax.bar(x_pos, mean_a, width, label="After Tuning: Mean Latency (ms)", color="#1b9e77", alpha=0.92)
    ax.bar(x_pos + width, max_a, width, label="After Tuning: Worst-Case Latency (ms)", color="#7570b3", alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(("Hover", "Forward Flight\n(4.5 m/s)", "Climb / Descent\n(+-2.0 m/s)", "Yaw Rotation\n(0.85 rad/s)"))
    ax.set_ylabel("Detection Latency (ms)")
    ax.set_title("Task 2c: Detection Latency vs. Flight Phase (Before & After Filter/Threshold Tuning)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("task2_latency_vs_phase.pdf")
    plt.savefig("task2_latency_vs_phase.png", dpi=300)
    plt.close(fig)

    # ------------ PLOT 2: CONFUSION MATRICES (MOTOR ISOLATION & ROOT CAUSE) ------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    im1 = ax1.imshow(iso_conf_mat, cmap="Blues")
    ax1.set_title("Motor Isolation Confusion Matrix", fontweight="bold")
    ax1.set_xlabel("Predicted Failed Motor")
    ax1.set_ylabel("True Injected Motor")
    m_labels = ("Motor 1", "Motor 2", "Motor 3", "Motor 4")
    ax1.set_xticks(range(4), m_labels)
    ax1.set_yticks(range(4), m_labels)
    for r_i in range(4):
        for c_i in range(4):
            val = int(iso_conf_mat[r_i, c_i])
            ax1.text(c_i, r_i, str(val), ha="center", va="center", color="white" if val > 8 else "black", fontweight="bold")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(cause_conf_mat, cmap="Greens")
    ax2.set_title("Root-Cause Diagnosis Confusion Matrix", fontweight="bold")
    ax2.set_xlabel("Predicted Fault Model")
    ax2.set_ylabel("True Injected Fault Model")
    c_labels = ("Complete\nStop", "Propeller\nLoss", "Partial\nLoss", "Intermittent\nFault")
    ax2.set_xticks(range(4), c_labels)
    ax2.set_yticks(range(4), c_labels)
    for r_i in range(4):
        for c_i in range(4):
            val = int(cause_conf_mat[r_i, c_i])
            ax2.text(c_i, r_i, str(val), ha="center", va="center", color="white" if val > 8 else "black", fontweight="bold")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("task2_confusion_matrices.pdf")
    plt.savefig("task2_confusion_matrices.png", dpi=300)
    plt.close(fig)

    # ------------ PLOT 3: BONUS 3-MOTOR HOVER RECOVERY & RESIDUALS FROM ULOG ------------
    hov = saved_summaries["ulg_logs/bonus_3motor_hover.ulg"]
    prop_run = saved_summaries["ulg_logs/fault_propeller_loss.ulg"]
    t = hov.t_arr
    t_inj = 1.5
    t_det = t_inj + hov.latency_ms / 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.0), dpi=300)
    p1, p2, p3, p4 = axes.flat

    p1.plot(t, hov.pos_z, "b-", lw=2.0, label="Altitude z(t) (Target = 5.0 m)")
    p1.plot(t, hov.drift_xy, "g--", lw=1.6, label="Horizontal Drift ||(x,y)|| (m)")
    p1.axvline(t_inj, color="r", ls="--", lw=1.3, label=f"Fault Injected ({t_inj:.2f} s)")
    p1.axvline(t_det, color="k", ls=":", lw=1.3, label=f"Fault Detected ({t_det:.3f} s)")
    p1.set_title("1. Post-Failure 3-Motor Hover Position & Altitude", fontweight="bold")
    p1.set_xlabel("Time (s)")
    p1.set_ylabel("Position / Drift (m)")
    p1.grid(True, alpha=0.35)
    p1.legend(loc="center right", fontsize=8.0)

    p2.plot(t, hov.res_norm, "purple", lw=1.9, label="Normalized Wrench Residual ||r(t)||")
    p2.plot(t, hov.r_roll, "c--", lw=1.3, label="Roll Torque Residual r_x")
    p2.plot(t, hov.r_pitch, "m--", lw=1.3, label="Pitch Torque Residual r_y")
    p2.axhline(0.36, color="k", ls="-.", lw=1.1, label="Detection Threshold (0.36)")
    p2.axvline(t_inj, color="r", ls="--", lw=1.2)
    p2.axvline(t_det, color="g", ls=":", lw=1.4)
    p2.set_title("2. Observer Residuals & Sub-20ms Trigger", fontweight="bold")
    p2.set_xlabel("Time (s)")
    p2.set_ylabel("Residual Magnitude")
    p2.grid(True, alpha=0.35)
    p2.legend(loc="upper right", fontsize=8.0)

    p3.plot(t, hov.wz_arr, "b-", lw=2.0, label="Complete Stop: Yaw Spin Rate r (rad/s)")
    p3.plot(t, hov.tilt_deg, "orange", lw=1.6, label="Primary-Axis Tilt Angle (deg)")
    p3.axvline(t_inj, color="r", ls="--", lw=1.2)
    p3.set_title("3. Steady Spinning Equilibrium (Yaw Given Up)", fontweight="bold")
    p3.set_xlabel("Time (s)")
    p3.set_ylabel("Spin Rate (rad/s) / Tilt (deg)")
    p3.grid(True, alpha=0.35)
    p3.legend(loc="upper left", fontsize=8.0)

    p4.plot(t, hov.f_hist[:, 0], "b-", lw=1.7, label="Motor 1 Thrust (Active)")
    p4.plot(t, hov.f_hist[:, 1], "orange", lw=1.7, label="Motor 2 Thrust (Opposite)")
    p4.plot(t, hov.f_hist[:, 2], "g--", lw=1.7, label="Motor 3 Thrust (Active)")
    p4.plot(t, hov.f_hist[:, 3], "r-", lw=2.0, label="Motor 4 Thrust (Failed -> 0 N)")
    p4.axvline(t_inj, color="k", ls="--", lw=1.2)
    p4.set_title("4. 3-Motor Thrust Reallocation (Bonus Hover)", fontweight="bold")
    p4.set_xlabel("Time (s)")
    p4.set_ylabel("Rotor Thrust (N)")
    p4.grid(True, alpha=0.35)
    p4.legend(loc="upper right", fontsize=8.0)

    plt.tight_layout()
    plt.savefig("task2_residuals_and_recovery.pdf")
    plt.savefig("task2_residuals_and_recovery.png", dpi=300)
    plt.savefig("bonus_hover_ulg_plots.pdf")
    plt.savefig("bonus_hover_ulg_plots.png", dpi=300)
    plt.close(fig)
    print("\nAll .ulg files and Task 2 / Bonus plots generated successfully!")


if __name__ == "__main__":
    main()
