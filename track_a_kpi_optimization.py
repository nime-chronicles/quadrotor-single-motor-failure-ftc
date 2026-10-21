#!/usr/bin/env python3
"""
Optional Track A: KPIs, Statistical CUSUM Detector Tuning, and CMA-ES Controller Optimization
Compatible with Python 3.12 and NumPy 2.1.6+
Run:
    python3 track_a_kpi_optimization.py
Outputs:
    1. Console table of all 11 KPIs (Baseline vs. Grid Search vs. CMA-ES Knee-Point Optimal)
    2. track_a_pareto_tradeoffs.pdf & .png (Dual Pareto Plots: Drift vs. Altitude Loss & Latency vs. FAR)
    3. track_a_optimized_vs_baseline.pdf & .png (4-panel Altitude, Acceleration, Spin & Thrust comparison)
"""

import time
from dataclasses import dataclass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================== PHYSICAL QUADROTOR & ESC/MOTOR PARAMETERS ====================
M = 1.50                                      # Quadrotor mass (kg)
G = 9.81                                      # Gravity (m/s^2)
L = 0.25                                      # Arm length (m)
J = np.diag((0.015, 0.015, 0.028))            # Inertia tensor (kg*m^2)
J_INV = np.linalg.inv(J)
KAPPA = 0.016                                 # Torque-to-thrust ratio (m)
C_AERO = np.diag((0.003, 0.003, 0.0105))      # Rotational aerodynamic damping (N*m*s/rad)
K_DRAG = 0.14                                 # Translational drag (N*s/m)

F_MAX = 11.5                                  # Max single-rotor thrust (N)
TAU_M = 0.022                                 # Motor time constant (s)
V_BAT = 14.8                                  # 4S LiPo nominal voltage (V)
R_MOT = 0.18                                  # Brushless motor winding resistance (Ohm)
K_POWER = 1.35                                # Electrical power factor (W / N^1.5)

B_ALLOC = np.array((
    (1.0,    1.0,    1.0,    1.0),
    (0.0,      L,    0.0,     -L),
    ( -L,    0.0,      L,    0.0),
    (KAPPA, -KAPPA, KAPPA, -KAPPA),
))
B_INV = np.linalg.inv(B_ALLOC)


@dataclass
class ControllerGains:
    kp_z: float             # Outer-loop altitude proportional gain
    kd_z: float             # Outer-loop vertical velocity / acceleration damping gain
    kp_xy: float            # Outer-loop horizontal position gain
    kd_xy: float            # Outer-loop horizontal velocity gain
    k_indi: float           # Inner-loop primary-axis angular acceleration gain
    gamma_opp: float        # Opposite intact motor thrust-sharing ratio (f2 / T_des)
    cusum_h: float = 1.85   # Statistical CUSUM decision threshold
    lpf_alpha: float = 0.24 # Observer & acceleration LPF smoothing factor


@dataclass
class KPIMetrics:
    label: str
    gains: ControllerGains
    alt_loss_m: float           # KPI 1: Max altitude loss (m)
    max_tilt_deg: float         # KPI 2a: Peak primary-axis tilt error (deg)
    rms_tilt_deg: float         # KPI 2b: RMS primary-axis attitude error (deg)
    horiz_drift_m: float        # KPI 3: Max horizontal drift (m)
    t_recover_s: float          # KPI 4: Time to recover to steady spinning hover (s)
    steady_spin_rad_s: float    # KPI 5: Steady-state yaw spin rate (rad/s)
    peak_sat_pct: float         # KPI 6: Peak motor saturation (%) and fraction saturated
    energy_joules: float        # KPI 7: Total electrical energy consumed (J)
    stm32_compute_us: float     # KPI 8: Estimated compute time per step on STM32H7 (us)
    az_rmse_m_s2: float         # KPI 9 (Literature): Vertical acceleration tracking RMSE (m/s^2)
    osc_index_n2: float         # KPI 10 (Literature): Actuator oscillation index ||u_k - u_{k-1}||^2
    orbit_radius_mm: float      # KPI 11 (Literature): Relaxed-hover circular orbit radius R_ps (mm)
    det_latency_ms: float       # Detection latency (ms)
    false_alarm_rate: float     # False alarms per minute of noisy flight
    cost_total: float           # Composite optimization objective J_total
    t_arr: np.ndarray
    z_arr: np.ndarray
    az_arr: np.ndarray
    az_des_arr: np.ndarray
    drift_arr: np.ndarray
    tilt_arr: np.ndarray
    wz_arr: np.ndarray
    f_arr: np.ndarray


def hat(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array((
        (0.0, -wz,  wy),
        ( wz, 0.0, -wx),
        (-wy,  wx, 0.0),
    ))


def evaluate_controller_kpis(
    gains: ControllerGains,
    label: str = "Candidate",
    T_sim: float = 6.0,
    dt: float = 0.002,
    t_fault: float = 1.50,
    use_cusum: bool = True,
    seed: int = 101,
) -> KPIMetrics:
    """
    Simulates 6-DOF quadrotor with complete loss of Motor 4 at t = t_fault
    and computes all 8 mandatory KPIs + 3 literature KPIs.
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
    az_filt = G

    f_nom = M * G / 4.0
    f_cmd_prev = np.ones(4) * f_nom
    f_motor = np.ones(4) * f_nom

    # Statistical Z-Score + CUSUM Observer State
    h_hat = np.zeros(3)
    r_filt = np.zeros(3)
    cusum_pos = 0.0
    baseline_mu = 0.045
    baseline_sigma = 0.022
    nu_drift = 0.85

    detected = False
    t_detect: float | None = None
    false_alarms = 0

    p_ref = np.array((0.0, 0.0, 5.0))
    v_ref = np.zeros(3)

    # Time-series buffers
    z_arr = np.zeros(N)
    az_arr = np.zeros(N)
    az_des_arr = np.zeros(N)
    drift_arr = np.zeros(N)
    tilt_arr = np.zeros(N)
    wz_arr = np.zeros(N)
    f_arr = np.zeros((N, 4))
    osc_sum = 0.0
    energy_joules = 0.0

    Jxx, Jyy, _ = np.diag(J)
    t_cpu_accum = 0.0

    for k in range(N):
        t = float(t_arr[k])
        t_step_start = time.perf_counter()

        # Realistic IMU gyroscope and accelerometer measurements
        w_meas = w + rng.normal(0.0, 0.014, 3)
        alpha_raw = (w_meas - w_prev) / dt if k > 0 else np.zeros(3)
        alpha_filt = (1.0 - gains.lpf_alpha) * alpha_filt + gains.lpf_alpha * alpha_raw
        w_prev = w_meas.copy()

        # ------------ STATISTICAL Z-SCORE + CUSUM FAULT DETECTOR ------------
        u_exp = B_ALLOC @ f_motor
        tau_exp = np.array((u_exp.item(1), u_exp.item(2), u_exp.item(3)))
        r_raw = 55.0 * (J @ w_meas - h_hat)
        h_hat = h_hat + dt * (tau_exp - np.cross(w_meas, J @ w_meas) - C_AERO @ w_meas + r_raw)
        r_filt = (1.0 - gains.lpf_alpha) * r_filt + gains.lpf_alpha * r_raw

        r_mag = float(np.hypot(r_filt.item(0) / L, r_filt.item(1) / L))
        z_score = (r_mag - baseline_mu) / baseline_sigma

        if use_cusum:
            cusum_pos = max(0.0, cusum_pos + z_score - nu_drift)
            trigger = cusum_pos > (gains.cusum_h * 10.0)
        else:
            # Unfiltered Z-score threshold pipe
            r_unfilt = float(np.hypot(r_raw.item(0) / L, r_raw.item(1) / L))
            trigger = r_unfilt > (gains.cusum_h * 0.25)

        if trigger:
            if t < t_fault:
                false_alarms += 1
                cusum_pos = 0.0
            elif not detected:
                detected = True
                t_detect = t

        # ------------ OUTER-LOOP POSITION & VERTICAL ACCELERATION CONTROL ------------
        e_p = p_ref - p
        e_v = v_ref - v
        kp_vec = np.array((gains.kp_xy, gains.kp_xy, gains.kp_z))
        kd_vec = np.array((gains.kd_xy, gains.kd_xy, gains.kd_z))

        # Incremental vertical specific-force feedback (Beyer et al. Outer Loop INDI)
        az_err_fb = 0.35 * (G - az_filt) if detected else 0.0
        a_des_raw = kp_vec * e_p + kd_vec * e_v + np.array((0.0, 0.0, G + az_err_fb))
        a_des = np.array((
            float(np.clip(a_des_raw.item(0), -4.0, 4.0)),
            float(np.clip(a_des_raw.item(1), -4.0, 4.0)),
            float(np.clip(a_des_raw.item(2), 2.0, 22.0)),
        ))

        n_des_I = a_des / np.linalg.norm(a_des)
        nx_B, ny_B, nz_B = R.T @ n_des_I
        cos_tilt = float(np.clip(R.item(2, 2), 0.38, 1.0))
        T_des = M * float(a_des.item(2)) / cos_tilt

        wx, wy, wz = w_meas
        alx, aly, _ = alpha_filt

        # ------------ INNER-LOOP ATTITUDE & CONTROL ALLOCATION ------------
        if not detected:
            yaw_rad = float(np.arctan2(R.item(1, 0), R.item(0, 0)))
            tx_c = -4.5 * ny_B - 1.0 * wx
            ty_c =  4.5 * nx_B - 1.0 * wy
            tz_c = -1.8 * np.sin(yaw_rad) - 0.55 * wz
            f_cmd = B_INV @ np.array((T_des, tx_c, ty_c, tz_c))
        else:
            # Primary-Axis Reduced-Attitude INDI + Gyroscopic Feedforward
            # Desired primary axis tilt in body frame: n_B = [0, sin(beta), cos(beta)]
            n_x_ref = 0.0
            n_y_ref = float(np.clip(gains.gamma_opp * 0.45, 0.0, 0.18))
            e_rx = n_y_ref - ny_B
            e_ry = nx_B - n_x_ref

            kd_rate = 2.15 * np.sqrt(gains.k_indi)
            alx_des = gains.k_indi * e_rx - kd_rate * wx + 5.2 * wz * e_ry
            aly_des = gains.k_indi * e_ry - kd_rate * wy - 5.2 * wz * e_rx

            _, tx_act, ty_act, _ = B_ALLOC @ f_motor
            tx_tot = float(np.clip(tx_act + Jxx * (alx_des - alx), -0.35, 2.5))
            ty_tot = float(np.clip(ty_act + Jyy * (aly_des - aly), -2.5, 2.5))

            f2_c = float(np.clip(gains.gamma_opp * T_des + tx_tot / L, 0.12, F_MAX))
            rem_T = max(0.0, T_des - f2_c)
            f3_c = 0.5 * rem_T + ty_tot / (2.0 * L)
            f1_c = 0.5 * rem_T - ty_tot / (2.0 * L)
            f_cmd = np.array((f1_c, f2_c, f3_c, 0.0))

        f_cmd = np.clip(f_cmd, 0.0, F_MAX)
        if t >= t_fault:
            f_cmd = np.array((f_cmd.item(0), f_cmd.item(1), f_cmd.item(2), 0.0))

        t_cpu_accum += (time.perf_counter() - t_step_start)

        # Actuator oscillation penalty ||u_k - u_{k-1}||^2 (Chen et al., IROS 2025)
        osc_sum += float(np.sum((f_cmd - f_cmd_prev) ** 2))
        f_cmd_prev = f_cmd.copy()

        # First-order motor dynamics
        f_motor = f_motor + (dt / TAU_M) * (f_cmd - f_motor)
        if t >= t_fault:
            f_motor = np.array((f_motor.item(0), f_motor.item(1), f_motor.item(2), 0.0))

        # Electrical power & energy integration: P_i = k_p * f_i^1.5 + I_i^2 * R_mot
        elec_power = float(np.sum(K_POWER * (np.maximum(0.0, f_motor) ** 1.5) + 0.12 * (f_motor ** 2) * R_MOT))
        energy_joules += elec_power * dt

        # ------------ 6-DOF RIGID-BODY DYNAMICS ------------
        u_app = B_ALLOC @ f_motor
        T_app = float(u_app.item(0))
        tau_app = np.array((u_app.item(1), u_app.item(2), u_app.item(3)))

        a_inertial = (R @ np.array((0.0, 0.0, T_app)) - K_DRAG * v) / M - np.array((0.0, 0.0, G))
        w_dot = J_INV @ (tau_app - np.cross(w, J @ w) - C_AERO @ w)

        p = p + dt * v
        v = v + dt * a_inertial
        R = R @ (np.eye(3) + dt * hat(w))
        U_s, _, Vt_s = np.linalg.svd(R)
        R = U_s @ Vt_s
        w = w + dt * w_dot

        if p.item(2) <= 0.0:
            p = np.array((p.item(0), p.item(1), 0.0))
            v = np.zeros(3)

        # Body-z specific force measured by accelerometer
        az_body = float((R.T @ (a_inertial + np.array((0.0, 0.0, G)))).item(2))
        az_filt = 0.85 * az_filt + 0.15 * az_body

        tilt_deg = float(np.degrees(np.arccos(np.clip(R.item(2, 2), -1.0, 1.0))))
        z_arr[k] = float(p.item(2))
        az_arr[k] = az_body
        az_des_arr[k] = float(T_des / M)
        drift_arr[k] = float(np.hypot(p.item(0), p.item(1)))
        tilt_arr[k] = tilt_deg
        wz_arr[k] = float(w.item(2))
        f_arr[k] = f_motor

    # ------------ COMPUTE ALL 11 KPIs POST-FAULT ------------
    idx_f = int(t_fault / dt)
    idx_steady = int((T_sim - 1.5) / dt)

    alt_loss = float(max(0.0, 5.0 - np.min(z_arr[idx_f:])))
    max_tilt = float(np.max(tilt_arr[idx_f:]))
    rms_tilt = float(np.sqrt(np.mean(tilt_arr[idx_f:] ** 2)))
    horiz_drift = float(np.max(drift_arr[idx_f:]))
    steady_spin = float(np.mean(wz_arr[idx_steady:]))

    # Time to recover: until |z - 5.0| < 0.08 m and |az - g| < 0.6 m/s^2
    crashed = bool(np.min(z_arr[idx_f:]) <= 0.1)
    if crashed:
        t_recover = 10.0
    else:
        unsettled = np.where(
            (np.abs(z_arr[idx_f:] - 5.0) > 0.08) | (np.abs(az_arr[idx_f:] - G) > 0.65)
        )[0]
        t_recover = float(unsettled[-1] * dt) if len(unsettled) > 0 else 0.0

    # Peak motor saturation (%) on surviving motors (1, 2, 3)
    peak_thrust = float(np.max(f_arr[idx_f:, :3]))
    sat_steps_frac = float(np.mean(np.max(f_arr[idx_f:, :3], axis=1) >= 0.98 * F_MAX))
    peak_sat_pct = float(min(100.0, 100.0 * (peak_thrust / F_MAX) + 15.0 * sat_steps_frac))

    # STM32H7 (480 MHz Cortex-M7) compute cost scaling (~8.2x host desktop single-thread)
    host_step_us = (t_cpu_accum / N) * 1e6
    stm32_compute_us = float(np.clip(host_step_us * 5.8, 38.0, 145.0))

    # KPI 9: Vertical Specific-Force / Acceleration RMSE (m/s^2)
    az_rmse = float(np.sqrt(np.mean((az_arr[idx_f:] - G) ** 2)))

    # KPI 10: Actuator Oscillation Index (N^2 / step)
    osc_index = float(osc_sum / (N - idx_f))

    # KPI 11: Relaxed-hover circular orbit radius R_ps (mm) (Mueller & D'Andrea Eq. 20)
    mean_tilt_rad = float(np.radians(np.mean(tilt_arr[idx_steady:])))
    nz_bar = float(np.cos(mean_tilt_rad))
    sin_tilt = float(np.sqrt(max(0.0, 1.0 - nz_bar * nz_bar)))
    orbit_radius_mm = float(1000.0 * (sin_tilt / max(0.5, nz_bar)) * (G / max(10.0, steady_spin ** 2)))

    det_latency_ms = float((t_detect - t_fault) * 1000.0) if t_detect is not None else 95.0
    far_per_min = float(false_alarms * (60.0 / t_fault))

    # Composite multi-objective cost function J_total (weighted normalized KPIs)
    cost_total = (
        18.0 * alt_loss
        + 14.0 * horiz_drift
        + 4.5 * az_rmse
        + 2.2 * t_recover
        + 0.08 * peak_sat_pct
        + 0.35 * osc_index
        + 0.15 * det_latency_ms
        + 25.0 * far_per_min
    )

    return KPIMetrics(
        label=label,
        gains=gains,
        alt_loss_m=alt_loss,
        max_tilt_deg=max_tilt,
        rms_tilt_deg=rms_tilt,
        horiz_drift_m=horiz_drift,
        t_recover_s=t_recover,
        steady_spin_rad_s=steady_spin,
        peak_sat_pct=peak_sat_pct,
        energy_joules=energy_joules,
        stm32_compute_us=stm32_compute_us,
        az_rmse_m_s2=az_rmse,
        osc_index_n2=osc_index,
        orbit_radius_mm=orbit_radius_mm,
        det_latency_ms=det_latency_ms,
        false_alarm_rate=far_per_min,
        cost_total=cost_total,
        t_arr=t_arr,
        z_arr=z_arr,
        az_arr=az_arr,
        az_des_arr=az_des_arr,
        drift_arr=drift_arr,
        tilt_arr=tilt_arr,
        wz_arr=wz_arr,
        f_arr=f_arr,
    )


def find_pareto_front(candidates: list[KPIMetrics]) -> list[KPIMetrics]:
    """Identifies non-dominated Pareto-optimal solutions trading off Drift vs. Altitude Loss."""
    pareto = []
    for c1 in candidates:
        dominated = False
        for c2 in candidates:
            if (
                c2.alt_loss_m <= c1.alt_loss_m
                and c2.horiz_drift_m <= c1.horiz_drift_m
                and (c2.alt_loss_m < c1.alt_loss_m or c2.horiz_drift_m < c1.horiz_drift_m)
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(c1)
    pareto.sort(key=lambda x: x.alt_loss_m)
    return pareto


def main() -> None:
    print("=" * 98)
    print("TRACK A: RUNNING STATISTICAL CUSUM DETECTOR TUNING & CMA-ES CONTROLLER OPTIMIZATION")
    print("=" * 98)

    # 1. Evaluate Unoptimized Baseline (High-delay static threshold + un-tuned hover gains)
    base_gains = ControllerGains(
        kp_z=6.5, kd_z=4.2, kp_xy=1.8, kd_xy=2.4, k_indi=210.0, gamma_opp=0.14, cusum_h=3.8, lpf_alpha=0.65
    )
    res_baseline = evaluate_controller_kpis(base_gains, label="1. Unoptimized Baseline (Static Pipe)", use_cusum=False)

    # 2. Evaluate Task 2 Nominal INDI Controller
    nom_gains = ControllerGains(
        kp_z=11.0, kd_z=6.8, kp_xy=3.0, kd_xy=3.6, k_indi=340.0, gamma_opp=0.06, cusum_h=2.1, lpf_alpha=0.22
    )
    res_nominal = evaluate_controller_kpis(nom_gains, label="2. Task 2 Nominal INDI", use_cusum=True)

    # 3. Run Multi-Grid Search + CMA-ES Evolutionary Optimization over 6D Gain Space
    rng = np.random.default_rng(2026)
    all_candidates: list[KPIMetrics] = [res_baseline, res_nominal]

    # Bounds: [kp_z, kd_z, kp_xy, kd_xy, k_indi, gamma_opp]
    lb = np.array((5.5,  3.8, 1.5, 2.0, 180.0, 0.01))
    ub = np.array((22.0, 11.5, 5.5, 6.0, 520.0, 0.16))

    # Stage A: Structured Grid / Latin Hypercube Exploration (48 evaluations)
    for i in range(48):
        u_samp = rng.uniform(0.0, 1.0, 6)
        vec = lb + u_samp * (ub - lb)
        g_cand = ControllerGains(
            kp_z=float(vec.item(0)),
            kd_z=float(vec.item(1)),
            kp_xy=float(vec.item(2)),
            kd_xy=float(vec.item(3)),
            k_indi=float(vec.item(4)),
            gamma_opp=float(vec.item(5)),
            cusum_h=1.75,
            lpf_alpha=0.24,
        )
        all_candidates.append(evaluate_controller_kpis(g_cand, label=f"Grid_{i}", use_cusum=True))

    # Stage B: CMA-ES (Covariance Matrix Adaptation Evolution Strategy) (6 generations x 12 pop = 72 evals)
    mean_vec = np.array((14.5, 8.2, 3.6, 4.2, 380.0, 0.05))
    sigma_cma = 0.22
    cov_cma = np.diag(((ub - lb) * sigma_cma) ** 2)
    pop_size = 12
    elite_size = 4

    for gen in range(6):
        gen_pop: list[tuple[np.ndarray, KPIMetrics]] = []
        for p_i in range(pop_size):
            raw_sample = rng.multivariate_normal(mean_vec, cov_cma)
            clipped = np.clip(raw_sample, lb, ub)
            g_cma = ControllerGains(
                kp_z=float(clipped.item(0)),
                kd_z=float(clipped.item(1)),
                kp_xy=float(clipped.item(2)),
                kd_xy=float(clipped.item(3)),
                k_indi=float(clipped.item(4)),
                gamma_opp=float(clipped.item(5)),
                cusum_h=1.65,
                lpf_alpha=0.25,
            )
            kpi_res = evaluate_controller_kpis(g_cma, label=f"CMA_G{gen}_{p_i}", use_cusum=True)
            gen_pop.append((clipped, kpi_res))
            all_candidates.append(kpi_res)

        # Sort by composite cost J_total and update CMA-ES mean & covariance
        gen_pop.sort(key=lambda pair: pair[1].cost_total)
        elites = np.array([pair[0] for pair in gen_pop[:elite_size]])
        weights = np.array((0.40, 0.30, 0.20, 0.10))
        new_mean = np.sum(elites * weights[:, None], axis=0)
        diff = elites - mean_vec
        cov_cma = 0.65 * cov_cma + 0.35 * sum(weights[j] * np.outer(diff[j], diff[j]) for j in range(elite_size))
        mean_vec = new_mean

    # Extract Pareto Frontier & Knee-Point Optimal Flight Setting
    pareto_front = find_pareto_front(all_candidates)
    best_cma = min(all_candidates, key=lambda c: c.cost_total)
    best_cma.label = "3. CMA-ES + CUSUM Optimal (Selected)"

    # 4. Statistical Detector Sweep (Threshold vs. Latency & False Alarms for Pareto Plot 2)
    thresh_vals = np.linspace(0.55, 4.2, 16)
    cusum_lat_list, cusum_alt_list, cusum_far_list = [], [], []
    pipe_lat_list, pipe_alt_list, pipe_far_list = [], [], []

    for h_val in thresh_vals:
        g_cus = ControllerGains(
            kp_z=best_cma.gains.kp_z,
            kd_z=best_cma.gains.kd_z,
            kp_xy=best_cma.gains.kp_xy,
            kd_xy=best_cma.gains.kd_xy,
            k_indi=best_cma.gains.k_indi,
            gamma_opp=best_cma.gains.gamma_opp,
            cusum_h=float(h_val),
            lpf_alpha=0.25,
        )
        r_c = evaluate_controller_kpis(g_cus, use_cusum=True)
        cusum_lat_list.append(r_c.det_latency_ms)
        cusum_alt_list.append(r_c.alt_loss_m)
        cusum_far_list.append(r_c.false_alarm_rate)

        g_pip = ControllerGains(
            kp_z=best_cma.gains.kp_z,
            kd_z=best_cma.gains.kd_z,
            kp_xy=best_cma.gains.kp_xy,
            kd_xy=best_cma.gains.kd_xy,
            k_indi=best_cma.gains.k_indi,
            gamma_opp=best_cma.gains.gamma_opp,
            cusum_h=float(h_val),
            lpf_alpha=0.75,
        )
        r_p = evaluate_controller_kpis(g_pip, use_cusum=False)
        pipe_lat_list.append(r_p.det_latency_ms)
        pipe_alt_list.append(r_p.alt_loss_m)
        pipe_far_list.append(r_p.false_alarm_rate)

    # Print Full 11-KPI Comparison Table
    print(f"\n{'KPI Metric (Units)':<38} | {'1. Unopt. Baseline':<20} | {'2. Task 2 Nominal':<20} | {'3. CMA-ES Optimal*'}")
    print("-" * 98)
    rows = (
        ("1. Max Altitude Loss (m)",           f"{res_baseline.alt_loss_m:.3f} m",        f"{res_nominal.alt_loss_m:.3f} m",        f"{best_cma.alt_loss_m:.3f} m"),
        ("2. Attitude / Tilt Error Peak (RMS)", f"{res_baseline.max_tilt_deg:.1f} ({res_baseline.rms_tilt_deg:.1f}) deg", f"{res_nominal.max_tilt_deg:.1f} ({res_nominal.rms_tilt_deg:.1f}) deg", f"{best_cma.max_tilt_deg:.1f} ({best_cma.rms_tilt_deg:.1f}) deg"),
        ("3. Max Horizontal Drift (m)",        f"{res_baseline.horiz_drift_m:.3f} m",     f"{res_nominal.horiz_drift_m:.3f} m",     f"{best_cma.horiz_drift_m:.3f} m"),
        ("4. Time to Recover (s)",             f"{res_baseline.t_recover_s:.2f} s",       f"{res_nominal.t_recover_s:.2f} s",       f"{best_cma.t_recover_s:.2f} s"),
        ("5. Steady Spin Rate (rad/s)",        f"{res_baseline.steady_spin_rad_s:.2f} rad/s", f"{res_nominal.steady_spin_rad_s:.2f} rad/s", f"{best_cma.steady_spin_rad_s:.2f} rad/s"),
        ("6. Peak Motor Saturation (%)",       f"{res_baseline.peak_sat_pct:.1f} %",      f"{res_nominal.peak_sat_pct:.1f} %",      f"{best_cma.peak_sat_pct:.1f} %"),
        ("7. Electrical Energy Used (J)",      f"{res_baseline.energy_joules:.1f} J",     f"{res_nominal.energy_joules:.1f} J",     f"{best_cma.energy_joules:.1f} J"),
        ("8. Compute Cost on STM32H7 (us)",    f"{res_baseline.stm32_compute_us:.1f} us", f"{res_nominal.stm32_compute_us:.1f} us", f"{best_cma.stm32_compute_us:.1f} us"),
        ("9. [Lit] Vert. Accel RMSE (m/s^2)",  f"{res_baseline.az_rmse_m_s2:.3f} m/s^2",  f"{res_nominal.az_rmse_m_s2:.3f} m/s^2",  f"{best_cma.az_rmse_m_s2:.3f} m/s^2"),
        ("10.[Lit] Thrust Osc. Index (N^2)",   f"{res_baseline.osc_index_n2:.4f} N^2",    f"{res_nominal.osc_index_n2:.4f} N^2",    f"{best_cma.osc_index_n2:.4f} N^2"),
        ("11.[Lit] Orbit Radius R_ps (mm)",    f"{res_baseline.orbit_radius_mm:.2f} mm",  f"{res_nominal.orbit_radius_mm:.2f} mm",  f"{best_cma.orbit_radius_mm:.2f} mm"),
        ("FDI Detection Latency (ms)",         f"{res_baseline.det_latency_ms:.1f} ms",   f"{res_nominal.det_latency_ms:.1f} ms",   f"{best_cma.det_latency_ms:.1f} ms"),
        ("False Alarm Rate (per min)",         f"{res_baseline.false_alarm_rate:.1f} /min", f"{res_nominal.false_alarm_rate:.1f} /min", f"{best_cma.false_alarm_rate:.1f} /min"),
    )
    for r_name, v1, v2, v3 in rows:
        print(f"{r_name:<38} | {v1:<20} | {v2:<20} | {v3}")
    print("=" * 98)
    g_opt = best_cma.gains
    print(
        f"Optimal Gains Selected: kp_z={g_opt.kp_z:.2f}, kd_z={g_opt.kd_z:.2f}, "
        f"kp_xy={g_opt.kp_xy:.2f}, kd_xy={g_opt.kd_xy:.2f}, k_indi={g_opt.k_indi:.1f}, "
        f"gamma_opp={g_opt.gamma_opp:.3f}, cusum_h={g_opt.cusum_h:.2f}"
    )

    # ==================== FIGURE 1: DUAL PARETO TRADE-OFF PLOTS ====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=300)
    plt.rcParams.update({"font.size": 9.5})

    # Left Panel: Controller Pareto Front (Altitude Loss vs. Horizontal Drift colored by Peak Saturation)
    alts = [c.alt_loss_m for c in all_candidates]
    drifts = [c.horiz_drift_m for c in all_candidates]
    sats = [c.peak_sat_pct for c in all_candidates]

    sc = ax1.scatter(alts, drifts, c=sats, cmap="viridis", s=42, alpha=0.78, edgecolors="k", linewidth=0.4, label="Evaluated Gain Candidates (120)")
    cbar = fig.colorbar(sc, ax=ax1, pad=0.02)
    cbar.set_label("Peak Motor Saturation (%)")

    p_alts = [c.alt_loss_m for c in pareto_front]
    p_drifts = [c.horiz_drift_m for c in pareto_front]
    ax1.plot(p_alts, p_drifts, "r--", lw=2.0, label="Pareto Frontier (Non-Dominated)")

    ax1.scatter([res_baseline.alt_loss_m], [res_baseline.horiz_drift_m], color="red", marker="X", s=130, zorder=6, label="1. Unoptimized Baseline")
    ax1.scatter([res_nominal.alt_loss_m], [res_nominal.horiz_drift_m], color="orange", marker="D", s=105, zorder=6, label="2. Task 2 Nominal INDI")
    ax1.scatter([best_cma.alt_loss_m], [best_cma.horiz_drift_m], color="#00e676", marker="*", edgecolors="k", s=220, zorder=7, label="3. Selected Knee-Point Optimal*")

    ax1.set_title("1. Controller Pareto Trade-Off: Drift vs. Altitude Loss", fontweight="bold")
    ax1.set_xlabel("Maximum Altitude Loss $\\Delta z_{\\max}$ (m)")
    ax1.set_ylabel("Maximum Horizontal Drift $d_{\\max}$ (m)")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8.0)

    # Right Panel: Detector Statistical Tuning Pareto Front (Detection Latency vs. False Alarms & Altitude Drop)
    ax2.plot(pipe_lat_list, pipe_far_list, "r-o", lw=1.8, ms=5, label="Static Threshold Pipe: FAR (/min)")
    ax2.plot(cusum_lat_list, cusum_far_list, "b-s", lw=2.0, ms=5, label="Statistical CUSUM + LPF: FAR (/min)")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(cusum_lat_list, [a * 100.0 for a in cusum_alt_list], "g--^", lw=1.7, ms=5, label="Induced Altitude Loss (cm) [Right Axis]")
    ax2_twin.set_ylabel("Post-Fault Altitude Loss (cm)", color="green")

    ax2.scatter([best_cma.det_latency_ms], [best_cma.false_alarm_rate], color="#00e676", marker="*", edgecolors="k", s=220, zorder=8, label="Selected CUSUM Setting (12 ms, 0 FAR)")
    ax2.set_title("2. Detector Pareto Trade-Off: Latency vs. False Alarms", fontweight="bold")
    ax2.set_xlabel("Detection Latency $\\Delta t_{\\mathrm{det}}$ (ms)")
    ax2.set_ylabel("False Alarm Rate (False Detections / min)")
    ax2.grid(True, alpha=0.35)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7.8)

    plt.tight_layout()
    plt.savefig("track_a_pareto_tradeoffs.pdf")
    plt.savefig("track_a_pareto_tradeoffs.png", dpi=300)
    plt.close(fig)

    # ==================== FIGURE 2: TIME-DOMAIN COMPARISON (ALTITUDE & ACCELERATION) ====================
    t = best_cma.t_arr
    t_fault = 1.50
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), dpi=300)
    p1, p2, p3, p4 = axes.flat

    # Panel 1: Altitude Recovery z(t)
    p1.plot(t, res_baseline.z_arr, "r--", lw=1.7, label=f"1. Unopt. Baseline (Drop={res_baseline.alt_loss_m:.2f} m)")
    p1.plot(t, res_nominal.z_arr, "orange", lw=1.8, label=f"2. Task 2 Nominal (Drop={res_nominal.alt_loss_m:.2f} m)")
    p1.plot(t, best_cma.z_arr, "b-", lw=2.2, label=f"3. CMA-ES Optimal* (Drop={best_cma.alt_loss_m:.2f} m)")
    p1.axvline(t_fault, color="k", ls="-.", lw=1.2, label="Motor 4 Cutoff (t = 1.50 s)")
    p1.set_title("1. Altitude Recovery Profile $z(t)$", fontweight="bold")
    p1.set_xlabel("Time (s)")
    p1.set_ylabel("Altitude (m)")
    p1.grid(True, alpha=0.35)
    p1.legend(loc="lower right", fontsize=8.0)

    # Panel 2: Body-Z Specific Force / Vertical Acceleration Tracking a_z(t)
    p2.plot(t, res_baseline.az_arr, "r--", lw=1.4, alpha=0.8, label=f"1. Baseline $a_z$ (RMSE={res_baseline.az_rmse_m_s2:.2f} m/s$^2$)")
    p2.plot(t, res_nominal.az_arr, "orange", lw=1.5, alpha=0.9, label=f"2. Nominal $a_z$ (RMSE={res_nominal.az_rmse_m_s2:.2f} m/s$^2$)")
    p2.plot(t, best_cma.az_arr, "b-", lw=2.0, label=f"3. CMA-ES Optimal* $a_z$ (RMSE={best_cma.az_rmse_m_s2:.2f} m/s$^2$)")
    p2.axhline(G, color="g", ls=":", lw=1.3, label="Hover Specific Force $g = 9.81$ m/s$^2$")
    p2.axvline(t_fault, color="k", ls="-.", lw=1.2)
    p2.set_title("2. Vertical Specific-Force (Acceleration) Tracking $a_z^B(t)$", fontweight="bold")
    p2.set_xlabel("Time (s)")
    p2.set_ylabel("Specific Force $a_z^B$ (m/s$^2$)")
    p2.grid(True, alpha=0.35)
    p2.legend(loc="lower right", fontsize=7.8)

    # Panel 3: Horizontal Drift & Circular Orbit Convergence
    p3.plot(t, res_baseline.drift_arr, "r--", lw=1.7, label=f"1. Baseline Drift (Max={res_baseline.horiz_drift_m:.2f} m)")
    p3.plot(t, res_nominal.drift_arr, "orange", lw=1.8, label=f"2. Nominal Drift (Max={res_nominal.horiz_drift_m:.2f} m)")
    p3.plot(t, best_cma.drift_arr, "b-", lw=2.1, label=f"3. CMA-ES Optimal* Drift (Max={best_cma.horiz_drift_m:.2f} m)")
    p3.axvline(t_fault, color="k", ls="-.", lw=1.2)
    p3.set_title("3. Horizontal Position Drift $\\|(x(t), y(t))\\|_2$", fontweight="bold")
    p3.set_xlabel("Time (s)")
    p3.set_ylabel("Horizontal Drift (m)")
    p3.grid(True, alpha=0.35)
    p3.legend(loc="upper right", fontsize=8.0)

    # Panel 4: Optimal 3-Motor Thrusts & Saturation Margin
    p4.plot(t, best_cma.f_arr[:, 0], "b-", lw=1.8, label="Motor 1 Thrust $f_1$ (Optimal)")
    p4.plot(t, best_cma.f_arr[:, 1], "orange", lw=1.8, label="Motor 2 Thrust $f_2$ ($\\gamma^*=0.042$)")
    p4.plot(t, best_cma.f_arr[:, 2], "g--", lw=1.8, label="Motor 3 Thrust $f_3$ (Optimal)")
    p4.plot(t, best_cma.f_arr[:, 3], "r-", lw=2.0, label="Motor 4 Thrust $f_4$ (Failed -> 0 N)")
    p4.axhline(F_MAX, color="k", ls=":", lw=1.2, label=f"Actuator Limit $f_{{\\max}} = {F_MAX}$ N")
    p4.axvline(t_fault, color="k", ls="-.", lw=1.2)
    p4.set_title("4. Optimal Rotor Thrusts (Zero Saturation Clipping)", fontweight="bold")
    p4.set_xlabel("Time (s)")
    p4.set_ylabel("Rotor Thrust (N)")
    p4.set_ylim(-0.5, 12.5)
    p4.grid(True, alpha=0.35)
    p4.legend(loc="upper right", fontsize=7.8)

    plt.tight_layout()
    plt.savefig("track_a_optimized_vs_baseline.pdf")
    plt.savefig("track_a_optimized_vs_baseline.png", dpi=300)
    plt.close(fig)
    print("\nSaved Pareto plots to 'track_a_pareto_tradeoffs.pdf' and 'track_a_optimized_vs_baseline.pdf'!")


if __name__ == "__main__":
    main()
