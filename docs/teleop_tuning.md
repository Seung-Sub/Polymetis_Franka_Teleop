# Teleop tuning guide — Vive ↔ Franka Cartesian impedance

This workspace controls the Franka Panda through Polymetis'
`HybridJointImpedanceController`, driven from a Cartesian-space setpoint
(`update_desired_ee_pose`). The Vive controller pose is mapped onto that
setpoint through `franka_vive_env.py`. Three things govern teleop "feel":

1. **Vive→robot motion mapping** — `pos_scale`, `rot_scale`, optional velocity
   clamp.
2. **Robot impedance gains** — `Kx_scale`, `Kxd_scale` multiplied onto the
   UMI baseline `Kx=[750,750,750, 15,15,15]`, `Kxd=[37,37,37, 2,2,2]`.
3. **Pipeline rates** — Vive poll 100 Hz, polymetis client 100 Hz (NUC
   watchdog ceiling), interpolation 200 Hz. These are fixed; do **not** raise
   `--teleop_frequency` above 100 unless you have re-bench'd the NUC.

---

## Presets (CLI: `--tuning_preset`)

| Preset    | pos_scale | rot_scale | Kx× | Kxd× | vel clamp | v_max | ω_max | Use case                            |
|-----------|-----------|-----------|-----|------|-----------|-------|-------|-------------------------------------|
| `coarse`  | 1.5       | 1.0       | 0.8 | 1.1  | off       | 2.0   | 2.5   | Large reach, transit, layout shots  |
| `normal`  | 1.0       | 1.0       | 1.0 | 1.0  | off       | 2.0   | 2.5   | Default — UMI baseline              |
| `precise` | 0.5       | 0.5       | 1.3 | 1.3  | on        | 0.4   | 1.0   | Fine inserts, alignment, contact    |
| `custom`  | —         | —         | —   | —    | —         | —     | —     | Override any subset; rest = normal  |

`--tuning_preset normal` is what Diffusion Policy / UMI ship with — start
there, then switch presets per task.

```bash
# Baseline (recommended first run)
bash bin/start_teleop.sh ~/Polymetis_Franka_Teleop/data/$(date +%Y%m%d_%H%M%S) \
     --tuning_preset normal

# Fine alignment
bash bin/start_teleop.sh ~/Polymetis_Franka_Teleop/data/$(date +%Y%m%d_%H%M%S) \
     --tuning_preset precise

# Mix & match
bash bin/start_teleop.sh ~/Polymetis_Franka_Teleop/data/$(date +%Y%m%d_%H%M%S) \
     --tuning_preset custom --pos_scale 0.7 --kx_scale 1.2 --velocity_clamp \
     --max_pos_velocity 0.6
```

The console prints the resolved values on startup:

```
[tuning] preset=precise | pos_scale=0.5 rot_scale=0.5 | Kx×=1.3 Kxd×=1.3 |
        vel_clamp=True v_max=0.4 m/s, ω_max=1.0 rad/s
```

---

## Symptom → knob

| You feel…                                  | Likely cause              | Try                                              |
|--------------------------------------------|---------------------------|--------------------------------------------------|
| Robot **lags** behind Vive                 | Stiffness too low         | `--kx_scale 1.2 → 1.5`                           |
| Robot **jitters / vibrates**               | Stiffness too high, or damping too low | `--kx_scale 0.8` or `--kxd_scale 1.3`            |
| Robot **overshoots** on stops              | Damping too low           | `--kxd_scale 1.3 → 1.5`                          |
| Robot **trips reflex / cartesian limit**   | Commanded velocity too high | `--velocity_clamp --max_pos_velocity 0.5`        |
| Hand moves too far for small Vive motion   | pos_scale too high        | `--pos_scale 0.7` (or `precise`)                 |
| Robot moves too little — wrist tires fast  | pos_scale too low         | `--pos_scale 1.3` (or `coarse`)                  |
| Rotation is too sensitive                  | rot_scale too high        | `--rot_scale 0.5`                                |
| Stutter / "pop" once a second              | NUC 1 s watchdog          | Lower `--teleop_frequency` to 100 (default)      |

---

## Safe ranges

These are the limits I've validated on KIST hardware. Outside them you risk
reflex trips, kinematic singularities, or drift that the impedance controller
can't recover from:

| Parameter         | Safe range       | Notes                                                  |
|-------------------|------------------|--------------------------------------------------------|
| `pos_scale`       | 0.3 – 1.8        | Above 2 → hand outruns reachable workspace fast        |
| `rot_scale`       | 0.3 – 1.5        | Above 1.5 the wrist easily flips through singularity   |
| `Kx_scale`        | 0.5 – 1.6        | UMI baseline 750 N/m; >1.6 starts shaking the table    |
| `Kxd_scale`       | 0.7 – 1.6        | Must rise *with* Kx — under-damped Kx is unstable      |
| `max_pos_velocity`| 0.2 – 2.0 m/s    | Reflex pos vel limit ≈ 2.5 m/s; leave headroom         |
| `max_rot_velocity`| 0.5 – 2.5 rad/s  | Reflex angular limit ≈ 2.6 rad/s                       |
| `teleop_frequency`| 100 (fixed)      | NUC `THRESHOLD_NS=1e9` watchdog; 200 Hz trips ~every 2 s |

---

## How tuning interacts with data quality

If you record demos for **diffusion policy / GR00T fine-tuning**, prefer
`normal` or `precise`. Both keep the Vive↔robot map close to identity, which
the policy will learn. `coarse`'s `pos_scale=1.5` distorts the mapping —
rollout policies trained on `normal` will be noticeably *under-shooting* if
you collected on `coarse`. Pick a preset per dataset and stay on it.

`Kx`/`Kxd` *do not* affect the recorded trajectory directly (we log the
desired EE pose, not the controller's transient response), but very low Kx
makes the recorded poses lag the Vive command the policy will be conditioned
on. Don't drop `Kx_scale` below 0.7 for data collection.

---

## Inside `franka_interpolation_controller.py`

For reference:

```python
# Defaults (UMI):
Kx_default  = np.array([750., 750., 750.,  15.,  15.,  15.])
Kxd_default = np.array([ 37.,  37.,  37.,   2.,   2.,   2.])

# Applied at start_cartesian_impedance:
robot.start_cartesian_impedance(
    Kx  = torch.from_numpy(Kx_default  * Kx_scale ).float(),
    Kxd = torch.from_numpy(Kxd_default * Kxd_scale).float(),
)
```

Per-axis tuning (e.g., stiff Z, compliant X/Y) is not exposed via CLI yet;
edit `franka_interpolation_controller.FrankaInterpolationController.run()` if
you need it.
