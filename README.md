# Robust Quadrotor Control Under Single-Motor Failure
**Inter IIT Tech Meet 15.0 — Mock Problem Statement Submission**

* **Submitted by:** Emin Philip Saji (`2501CB04`)
* **Institute:** Indian Institute of Technology Patna
* **Target Stack:** Ubuntu 22.04 / 24.04 LTS, Python 3.12+, PX4 Autopilot (SITL), Gazebo Harmonic, MuJoCo 3.9+, PlotJuggler 3.17+

---

## 1. Project Overview
When one of a quadrotor's four motors fails mid-flight, the vehicle loses the ability to control roll, pitch, yaw, and altitude simultaneously. A standard 4-motor controller (such as stock PX4 PID) tries to hold yaw heading, saturates the remaining actuators, and crashes within seconds.

This repository provides a complete, reproducible **Fault Detection, Isolation, and Root-Cause Diagnosis (FDI)** and **Fault-Tolerant Control (FTC)** pipeline that:
1. **Injects 4 physical actuator fault models** from a script (`complete_stop`, `propeller_loss`, `partial_loss` $30\text{--}80\%$, and `intermittent` loose connector) on any motor across all flight phases.
2. **Detects, isolates, and diagnoses the root cause** in $\sim 12\text{--}16\text{ ms}$ with **zero false alarms** using a drag-compensated Generalized Momentum & Specific-Force Residual Observer fused with ESC RPM telemetry and Page's CUSUM test.
3. **Recovers into a stable 3-motor spinning hover** by relinquishing yaw angle control and aligning the body-fixed **Primary Axis ($\mathbf{n}$)** via a Prioritized Incremental Nonlinear Dynamic Inversion (INDI) controller tuned with CMA-ES.
4. **Logs all telemetry natively into binary PX4 `.ulg` files** inside `ulg_logs/` with custom uORB topics and embedded `[FAULT_INJECT]` and `[FAULT_DETECT]` timestamps inspectable in `pyulog` and **PlotJuggler**.

---

## 2. Repository File Structure

```text
.
├── Report.pdf                            # Final Compiled Technical Report
├── README.md                             # Project documentation & reproducibility guide
├── requirements.txt                      # Pinned Python dependencies
│
├── run_benchmark.py                      # Task 1D: 6-DOF baseline benchmark (Stock PID vs. LQR vs. INDI)
├── task2_pipeline_and_ulog.py            # Task 2 (2a-2c): Multi-phase FDI benchmark & binary .ulg generator
├── track_a_kpi_optimization.py           # Optional Track A: Page's CUSUM + CMA-ES optimization over 11 KPIs
├── run_realistic_drone_sim.py            # Bonus: 3D MuJoCo side-by-side visual simulator + live .ulg recorder
│
├── Bonus Track - Simulation.webm         # Bonus: Screen recording of 3D takeoff, fault, crash vs. 3-motor hover
├── FTC_Companies_Labs_OpenSource.xlsx    # Bonus: Curated Excel database of global startups, labs & public code
├── Image.png                             # Step 0: Screenshot of verified Ubuntu PX4/Gazebo/MuJoCo/PlotJuggler setup
│
├── ulg_logs/                             # Native Binary PX4 ULog (.ulg) Flight Logs
│   ├── fault_complete_stop.ulg           # Task 2c: Complete motor/ESC stop on Motor 4
│   ├── fault_propeller_loss.ulg          # Task 2c: Propeller blade detachment on Motor 2 (high RPM, 0 thrust)
│   ├── fault_partial_loss.ulg            # Task 2c: Partial actuator degradation (45% effectiveness) on Motor 3
│   ├── fault_intermittent.ulg            # Task 2c: Intermittent connector toggling fault on Motor 1
│   ├── bonus_3motor_hover.ulg            # Bonus: Complete 3-motor spinning hover recovery log
│   └── live_3d_simulation.ulg            # Bonus: Live-streamed .ulg from the 3D realistic drone simulation
│
└── Generated Plots (.pdf)
    ├── benchmark_results.pdf             # Task 1D 4-panel baseline comparison plot
    ├── task2_latency_vs_phase.pdf        # Task 2c detection latency across flight phases
    ├── task2_confusion_matrices.pdf      # Task 2c Motor Isolation & Root-Cause confusion matrices
    ├── task2_residuals_and_recovery.pdf  # Task 2 observer residuals and 3-motor recovery plot
    ├── track_a_pareto_tradeoffs.pdf      # Track A dual Pareto frontier trade-off plots
    ├── track_a_optimized_vs_baseline.pdf # Track A optimized vs. baseline time-domain comparison
    ├── live_ulg_simulation_plots.pdf     # Phase-shaded telemetry from live 3D simulation .ulg
    └── bonus_hover_ulg_plots.pdf         # Bonus 3-motor hover recovery plots from .ulg
