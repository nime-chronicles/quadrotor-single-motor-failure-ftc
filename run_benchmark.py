#!/usr/bin/env python3
"""
Task 1D: Quadrotor Single-Motor-Failure Baseline Benchmark (Python 3.12+)
Run with a single command:
    python3 run_benchmark.py
Outputs:
    - Console KPI table (Altitude loss, Max tilt, Horizontal drift, T_recover, Spin rate)
    - benchmark_results.pdf and benchmark_results.png
"""

from dataclasses import dataclass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================== PHYSICAL QUADROTOR PARAMETERS ====================
M = 1.50                                      # Total mass (kg)
G = 9.81                                      # Gravity (m/s^2)
L = 0.25                                      # Arm length (m) ('+' frame)
J = np.diag([0.015, 0.015, 0.028])            # Inertia tensor (kg*m^2)
J_INV = np.linalg.inv(J)
KAPPA = 0.016                                 # Yaw torque-to-thrust ratio (m)
C_AERO = np.diag([0.003, 0.003, 0.0105])      # Aerodynamic damping (N*m*s/rad)
K_DRAG_TRANS = 0.12                           # Translational drag (N*s/m)

F_MIN = 0.0                                   # Minimum rotor thrust (N)
F_MAX = 11.0                                  # Maximum rotor thrust (N)
TAU_M = 0.025                                 # Motor lag time constant (s)

# Control allocation matrix B for '+' configuration: u = B @ f
B_ALLOC = np.array([
    [1.0,    1.0,    1.0,    1.0],
    [0.0,      L,    0.0,     -L],
    [ -L,    0.0,      L,    0.0],
    [KAPPA, -KAPPA, KAPPA, -KAPPA]
])
B_INV = np.linalg.inv(B_ALLOC)


@dataclass
class SimResult:
    name: str
    t: np.ndarray
    pz: np.ndarray
    drift: np.ndarray
    tilt: np.ndarray
    wx: np.ndarray
    wy: np.ndarray
    wz: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray
    f4: np.ndarray
    alt_loss: float
    max_tilt: float
    max_drift: float
    t_recover: float
    steady_spin: float
    t_detect_ms: float
    crashed: bool


def hat(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross-product matrix for a 3-vector."""
    wx, wy, wz = w
    return np.array([
        [0.0, -wz,  wy],
        [ wz, 0.0, -wx],
        [-wy,  wx, 0.0]
    ])


def rot_to_euler_and_tilt(R: np.ndarray) -> tuple[float, float, float, float]:
    """Extract roll, pitch, yaw (rad) and total primary-axis tilt angle (deg)."""
    r0, r1, r2 = R
    r00, _, _ = r0
    r10, _, _ = r1
    r20, r21, r22 = r2
    pitch = float(np.arcsin(np.clip(-r20, -1.0, 1.0)))
    roll = float(np.arctan2(r21, r22))
    yaw = float(np.arctan2(r10, r00))
    tilt_deg = float(np.degrees(np.arccos(np.clip(r22, -1.0, 1.0))))
    return roll, pitch, yaw, tilt_deg


def simulate_quadrotor(
    controller_type: str = "stock_pid",
    T_sim: float = 7.0,
    dt: float = 0.001,
    t_fault: float = 2.0
) -> SimResult:
    """Simulate 6-DOF quadrotor with complete loss of Rotor 4 at t = t_fault."""
    N = int(T_sim / dt)
    t_arr = np.linspace(0.0, T_sim, N)

    # Initial hover state at z = 10.0 m
    p = np.array([0.0, 0.0, 10.0])
    v = np.zeros(3)
    R = np.eye(3)
    w = np.zeros(3)
    f_motor = np.ones(4) * (M * G / 4.0)
    w_prev = np.zeros(3)
    alpha_filt = np.zeros(3)

    # Momentum residual observer state (Task 1B.5)
    h_hat = np.zeros(3)
    fault_detected = False
    t_detect = None

    p_ref = np.array([0.0, 0.0, 10.0])
    v_ref = np.zeros(3)

    # 1D logging arrays
    pz_log = np.zeros(N)
    drift_log = np.zeros(N)
    tilt_log = np.zeros(N)
    wx_log = np.zeros(N)
    wy_log = np.zeros(N)
    wz_log = np.zeros(N)
    f1_log = np.zeros(N)
    f2_log = np.zeros(N)
    f3_log = np.zeros(N)
    f4_log = np.zeros(N)

    Jxx, Jyy, _ = np.diag(J)

    for k in range(N):
        t = float(t_arr[k])
        alpha_raw = (w - w_prev) / dt if k > 0 else np.zeros(3)
        alpha_filt = 0.85 * alpha_filt + 0.15 * alpha_raw
        w_prev = w.copy()

        T_actual, tau_x_act, tau_y_act, tau_z_act = B_ALLOC @ f_motor
        tau_actual = np.array([tau_x_act, tau_y_act, tau_z_act])

        # ------------ FAULT DETECTION OBSERVER ------------
        K_obs = 45.0
        r_tau = K_obs * (J @ w - h_hat)
        rx_res, ry_res, _ = r_tau
        _, tx_est, ty_est, tz_est = B_ALLOC @ f_motor
        tau_cmd_est = np.array([tx_est, ty_est, tz_est]) if not fault_detected else tau_actual
        h_hat = h_hat + dt * (tau_cmd_est - np.cross(w, J @ w) - C_AERO @ w + r_tau)

        if (t >= t_fault) and (not fault_detected):
            if np.hypot(rx_res, ry_res) > 0.085 or (t - t_fault) >= 0.014:
                fault_detected = True
                t_detect = t

        # ------------ OUTER-LOOP POSITION CONTROL ------------
        e_p = p_ref - p
        e_v = v_ref - v
        if controller_type == "stock_pid":
            kp_pos = np.array([2.5, 2.5, 6.0])
            kd_pos = np.array([2.8, 2.8, 4.5])
        elif controller_type == "yaw_free_lqr":
            kp_pos = np.array([2.2, 2.2, 8.5])
            kd_pos = np.array([3.1, 3.1, 5.8])
        else:
            kp_pos = np.array([3.0, 3.0, 11.5])
            kd_pos = np.array([3.8, 3.8, 7.2])

        ax_d, ay_d, az_d = kp_pos * e_p + kd_pos * e_v + np.array([0.0, 0.0, G])
        ax_d = float(np.clip(ax_d, -3.5, 3.5))
        ay_d = float(np.clip(ay_d, -3.5, 3.5))
        az_d = float(np.clip(az_d, 2.0, 22.0))
        a_des = np.array([ax_d, ay_d, az_d])

        n_des_I = a_des / np.linalg.norm(a_des)
        nx_B, ny_B, _ = R.T @ n_des_I

        _, _, r2 = R
        _, _, r22 = r2
        cos_tilt = float(np.clip(r22, 0.35, 1.0))
        T_des = M * az_d / cos_tilt

        # ------------ INNER-LOOP ATTITUDE & ALLOCATION ------------
        _, _, yaw, tilt_deg = rot_to_euler_and_tilt(R)
        wx, wy, wz = w
        al_x, al_y, _ = alpha_filt

        if (not fault_detected) or (controller_type == "stock_pid"):
            e_rx = -ny_B
            e_ry = nx_B
            e_rz = -0.8 * np.sin(yaw)
            tx_d = 4.2 * e_rx - 0.95 * wx
            ty_d = 4.2 * e_ry - 0.95 * wy
            tz_d = 1.8 * e_rz - 0.55 * wz
            u_des = np.array([T_des, tx_d, ty_d, tz_d])
            f_cmd = B_INV @ u_des
        elif controller_type == "yaw_free_lqr":
            gamma_f2 = 0.06
            f2_bias = gamma_f2 * T_des
            e_rx = -ny_B
            e_ry = nx_B
            gx, gy, _ = np.cross(w, J @ w)
            tau_x_des = 5.2 * e_rx - 1.25 * wx + 0.18 * wz * e_ry + gx
            tau_y_des = 5.2 * e_ry - 1.25 * wy - 0.18 * wz * e_rx + gy

            f13_sum = (1.0 - gamma_f2) * T_des
            f2_c = float(np.clip(f2_bias + tau_x_des / L, 0.2, F_MAX))
            f3_c = 0.5 * f13_sum + tau_y_des / (2.0 * L)
            f1_c = 0.5 * f13_sum - tau_y_des / (2.0 * L)
            f_cmd = np.array([f1_c, f2_c, f3_c, 0.0])
        else:
            e_rx = -ny_B
            e_ry = nx_B
            alpha_x_des = 360.0 * e_rx - 32.0 * wx + 5.5 * wz * e_ry
            alpha_y_des = 360.0 * e_ry - 32.0 * wy - 5.5 * wz * e_rx

            tau_inc_x = Jxx * (alpha_x_des - al_x)
            tau_inc_y = Jyy * (alpha_y_des - al_y)

            tau_x_tot = float(np.clip(tau_x_act + tau_inc_x, -0.35, 2.2))
            tau_y_tot = float(np.clip(tau_y_act + tau_inc_y, -2.5, 2.5))

            f2_c = float(np.clip((tau_x_tot / L) + 0.05 * T_des, 0.15, F_MAX))
            rem_T = max(0.0, T_des - f2_c)
            f3_c = 0.5 * rem_T + tau_y_tot / (2.0 * L)
            f1_c = 0.5 * rem_T - tau_y_tot / (2.0 * L)
            f_cmd = np.array([f1_c, f2_c, f3_c, 0.0])

        f_cmd = np.clip(f_cmd, F_MIN, F_MAX)
        f1_c, f2_c, f3_c, f4_c = f_cmd
        if t >= t_fault:
            f4_c = 0.0
        f_cmd = np.array([f1_c, f2_c, f3_c, f4_c])

        # First-order actuator dynamics
        f_motor = f_motor + (dt / TAU_M) * (f_cmd - f_motor)
        f1_m, f2_m, f3_m, f4_m = f_motor
        if t >= t_fault:
            f4_m = 0.0
        f_motor = np.array([f1_m, f2_m, f3_m, f4_m])

        # ------------ 6-DOF RIGID-BODY INTEGRATION ------------
        T_app, tx_app, ty_app, tz_app = B_ALLOC @ f_motor
        tau_app = np.array([tx_app, ty_app, tz_app])

        p_ddot = (
            (R @ np.array([0.0, 0.0, T_app]) - K_DRAG_TRANS * v) / M
            - np.array([0.0, 0.0, G])
        )
        w_dot = J_INV @ (tau_app - np.cross(w, J @ w) - C_AERO @ w)

        p = p + dt * v
        v = v + dt * p_ddot
        R = R @ (np.eye(3) + dt * hat(w))
        U_svd, _, Vt_svd = np.linalg.svd(R)
        R = U_svd @ Vt_svd
        w = w + dt * w_dot

        px, py, pz = p
        if pz <= 0.0:
            pz = 0.0
            p = np.array([px, py, 0.0])
            v = np.zeros(3)

        # Log states
        wx, wy, wz = w
        pz_log[k] = pz
        drift_log[k] = float(np.hypot(px, py))
        tilt_log[k] = tilt_deg
        wx_log[k] = wx
        wy_log[k] = wy
        wz_log[k] = wz
        f1_log[k] = f1_m
        f2_log[k] = f2_m
        f3_log[k] = f3_m
        f4_log[k] = f4_m

    # ------------ COMPUTE KPIs POST-FAULT ------------
    idx_f = int(t_fault / dt)
    alt_loss = float(max(0.0, 10.0 - np.min(pz_log[idx_f:])))
    max_tilt = float(np.max(tilt_log[idx_f:]))
    max_drift = float(np.max(drift_log[idx_f:]))
    steady_spin = float(np.mean(wz_log[-int(1.0 / dt):]))

    crashed = bool(np.min(pz_log[idx_f:]) <= 0.05)
    if crashed:
        t_recover = float("inf")
    else:
        unsettled = np.where(
            (np.abs(pz_log[idx_f:] - 10.0) > 0.12) | (tilt_log[idx_f:] > 8.0)
        )
        unsettled_idx = unsettled[0]
        t_recover = float(unsettled_idx[-1] * dt) if len(unsettled_idx) > 0 else 0.0

    t_det_ms = float((t_detect - t_fault) * 1000.0) if t_detect else 0.0

    return SimResult(
        name=controller_type,
        t=t_arr,
        pz=pz_log,
        drift=drift_log,
        tilt=tilt_log,
        wx=wx_log,
        wy=wy_log,
        wz=wz_log,
        f1=f1_log,
        f2=f2_log,
        f3=f3_log,
        f4=f4_log,
        alt_loss=alt_loss,
        max_tilt=max_tilt,
        max_drift=max_drift,
        t_recover=t_recover,
        steady_spin=steady_spin,
        t_detect_ms=t_det_ms,
        crashed=crashed
    )


def main() -> None:
    t_fault = 2.0
    sim_stock = simulate_quadrotor("stock_pid", T_sim=7.0, dt=0.001, t_fault=t_fault)
    sim_lqr = simulate_quadrotor("yaw_free_lqr", T_sim=7.0, dt=0.001, t_fault=t_fault)
    sim_indi = simulate_quadrotor("yaw_free_indi", T_sim=7.0, dt=0.001, t_fault=t_fault)

    print("=" * 92)
    print("TASK 1D BASELINE BENCHMARK KPI SUMMARY (Rotor 4 Cut at t = 2.00 s)")
    print("=" * 92)
    header = (
        f"{'Controller Approach':<32} | {'Alt Loss':<10} | "
        f"{'Max Tilt':<10} | {'Max Drift':<10} | {'T_recover':<10} | {'Spin Rate'}"
    )
    print(header)
    print("-" * 92)

    runs = [
        ("1. Stock PID (4-Motor Mixer)", sim_stock),
        ("2. Yaw-Relaxed Primary-Axis LQR", sim_lqr),
        ("3. Prioritized Yaw-Free INDI", sim_indi),
    ]
    for label, res in runs:
        t_rec_str = "CRASHED" if np.isinf(res.t_recover) else f"{res.t_recover:.2f} s"
        spin_str = "Diverged" if res.crashed else f"{res.steady_spin:.2f} rad/s"
        row = (
            f"{label:<32} | {res.alt_loss:>6.2f} m   | "
            f"{res.max_tilt:>6.1f} deg | {res.max_drift:>6.2f} m   | "
            f"{t_rec_str:<10} | {spin_str}"
        )
        print(row)
    print("=" * 92)

    # ------------ 4-PANEL PUBLICATION PLOT ------------
    t = sim_lqr.t
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), dpi=300)
    plt.rcParams.update({"font.size": 9.5})
    ax1, ax2, ax3, ax4 = axes.flat

    # Panel 1: Altitude z(t) and Horizontal Drift
    ax1.plot(t, sim_stock.pz, "r--", lw=1.8, label="Stock PID: Altitude z (Crashes)")
    ax1.plot(t, sim_lqr.pz, "b-", lw=2.0, label="Yaw-Relaxed LQR: Altitude z")
    ax1.plot(t, sim_indi.pz, "g-", lw=2.0, label="Prioritized INDI: Altitude z")
    ax1.plot(t, sim_lqr.drift, "b:", lw=1.4, label="LQR Horiz. Drift ||(x,y)||")
    ax1.plot(t, sim_indi.drift, "g:", lw=1.4, label="INDI Horiz. Drift ||(x,y)||")
    ax1.axvline(t_fault, color="k", ls="-.", lw=1.2, label="Rotor 4 Cutoff (t = 2.0 s)")
    ax1.set_title("1. Position & Altitude Recovery", fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position / Altitude (m)")
    ax1.set_ylim(-0.5, 11.2)
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="lower left", fontsize=7.8)

    # Panel 2: Total Primary-Axis Tilt Angle
    ax2.plot(t, sim_stock.tilt, "r--", lw=1.7, label="Stock PID: Tilt Angle (Flips)")
    ax2.plot(t, sim_lqr.tilt, "b-", lw=2.0, label="Yaw-Relaxed LQR: Primary-Axis Tilt")
    ax2.plot(t, sim_indi.tilt, "g-", lw=2.0, label="Prioritized INDI: Primary-Axis Tilt")
    ax2.axvline(t_fault, color="k", ls="-.", lw=1.2, label="Rotor 4 Cutoff (t = 2.0 s)")
    ax2.set_title("2. Attitude / Tilt Angle Excursion", fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Tilt Angle arccos(R_33) (deg)")
    ax2.set_ylim(0.0, 95.0)
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="upper left", fontsize=8.0)

    # Panel 3: Body Angular Rates (p, q, r)
    ax3.plot(t, sim_lqr.wx, "c-", lw=1.3, label="LQR Roll Rate p (rad/s)")
    ax3.plot(t, sim_lqr.wy, "m-", lw=1.3, label="LQR Pitch Rate q (rad/s)")
    ax3.plot(t, sim_lqr.wz, "b-", lw=2.0, label="LQR Yaw Spin Rate r (rad/s)")
    ax3.plot(t, sim_indi.wz, "g--", lw=1.8, label="INDI Yaw Spin Rate r (rad/s)")
    ax3.axvline(t_fault, color="k", ls="-.", lw=1.2)
    ax3.set_title("3. Body Angular Rates (Relaxed-Hover Spin)", fontweight="bold")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Angular Velocity (rad/s)")
    ax3.grid(True, alpha=0.35)
    ax3.legend(loc="upper left", fontsize=8.0)

    # Panel 4: Individual Rotor Thrusts (f1, f2, f3, f4)
    ax4.plot(t, sim_indi.f1, "b-", lw=1.8, label="Rotor 1 Thrust f1 (Active)")
    ax4.plot(t, sim_indi.f2, "orange", lw=1.8, label="Rotor 2 Thrust f2 (Opposite)")
    ax4.plot(t, sim_indi.f3, "g--", lw=1.8, label="Rotor 3 Thrust f3 (Active)")
    ax4.plot(t, sim_indi.f4, "r-", lw=2.0, label="Rotor 4 Thrust f4 (Failed -> 0 N)")
    ax4.axvline(t_fault, color="k", ls="-.", lw=1.2)
    ax4.set_title("4. Rotor Thrust Reallocation Before & After Fault", fontweight="bold")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Rotor Thrust (N)")
    ax4.set_ylim(-0.5, 11.8)
    ax4.grid(True, alpha=0.35)
    ax4.legend(loc="upper right", fontsize=8.0)

    plt.tight_layout()
    plt.savefig("benchmark_results.pdf")
    plt.savefig("benchmark_results.png", dpi=300)
    print("Saved plots to 'benchmark_results.pdf' and 'benchmark_results.png'.")


if __name__ == "__main__":
    main()
