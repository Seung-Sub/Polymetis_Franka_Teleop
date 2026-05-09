# Troubleshooting catalog

Issues we hit during the KIST bring-up + the fixes we ended up with.
Index by symptom; each row links into the relevant install phase or
code path.

## RT / Polymetis stability

### #1 — `motion aborted by reflex! [communication_constraints_violation] success_rate: 0.79`

Polymetis prints this line and the policy disengages within seconds of any motion command.

**Cause:** the Franka NIC IRQ ended up on a P-core (default round-robin), where it competes with GUI processes (gnome-shell, Firefox, etc.). The 1 kHz inner loop sees jitter > 1 ms and trips the libfranka `communication_constraints_violation` threshold.

**Fix:** Phase D — `franka_rt_apply.sh` re-pins the NIC IRQs to E-cores 12-15 at boot. Verify with:

```bash
for irq in $(grep enp86s0 /proc/interrupts | awk '{print $1}' | tr -d ':'); do
    echo "IRQ $irq: $(cat /proc/irq/$irq/smp_affinity_list)"
done
```

Each line should show 12, 13, 14, or 15. If they show 0-5, `franka-rt-tune.service` did not run — `systemctl status franka-rt-tune` and check why.

### #2 — `success_rate: 0.24` (worse than #1)

**Cause:** even when NIC IRQs are pinned, the Polymetis RT threads themselves did not migrate to the isolated cores 6,7. `isolcpus` only *prevents* the scheduler from putting tasks on those cores by default; it does not *pull* tasks in. Without `taskset`, the RT threads spawn on whatever P-core was free at exec time and stay there, sharing it with GUI.

**Fix:** Phase D — `start_franka_arm.sh` schedules `franka_pin_helper.sh` 5 seconds after `launch_robot.py` spawns, which `taskset -cpa 6,7`s every TID of `run_server`, `franka_panda_client`, and `franka_hand_client`. The wrapper prints

```
[arm pinner] cores 6,7 pin applied (details: tail ~/.franka_logs/franka_pin_arm.log)
```

If this line never appears: the sudoers drop-in is missing. `sudo bash install/install_nuc.sh` again.

### #3 — Recovery cascade (8–12 reflex trips per minute) during teleop

**Cause:** when the Vive controller reaches a pose the Franka can't kinematically achieve, `franka_interpolation_controller.py`'s IK call returns `success=False` silently. With no `update_desired_joint_positions` call going out for >1 s, the Polymetis 1-second watchdog kills the impedance policy. The recovery handler restarts impedance, but if the operator keeps pushing toward the unreachable target, this loops.

**Fix:** the controller in this repo (`franka_interpolation_controller.py`) caches the last successful joint target and replays it on IK failure, keeping the watchdog fed:

```python
if not success:
    self._ik_fail_streak += 1
    if self._last_good_joint_target is not None:
        self.robot.update_desired_joint_positions(self._last_good_joint_target)
    return
```

Plus the recovery branch sleeps 1.0 s (was 0.5 s) after `start_cartesian_impedance` to let the policy settle before resuming control. With both fixes a 60-minute teleop session has 0 recoveries.

### #4 — NUC `start_franka_arm.sh` runs, no `[arm pinner]` line, polymetis seems alive but the robot stutters under load

**Cause:** sudoers drop-in `/etc/sudoers.d/franka_rt` not in place — `sudo -n /usr/local/sbin/franka_pin_helper.sh` returns "a password is required" silently because of the `>` redirect in `start_franka_arm.sh`.

**Fix:** `sudo bash install/install_nuc.sh` (which installs the drop-in with `visudo -c` validation).

### #5 — NUC arm server "frozen" after 30+ minutes of uptime

**Symptom:** clients can connect to `:50051`, get joint state, but motion commands silently no-op.

**Cause:** rare libfranka state corruption. Not yet root-caused, doesn't happen with current code in <60 min sessions.

**Fix:** restart the arm server.

```bash
ssh kist@192.168.1.12 'pkill -INT launch_robot; sleep 2; sudo bash /usr/local/sbin/start_franka_arm.sh' &
```

---

## Polymetis client (pro4000 side)

### #6 — `libtorchscript_pinocchio.so: cannot open shared object`

**Cause:** the NUC built Polymetis (Phase C); the pro4000 side imports `polymetis` but the .so files only exist on the NUC.

**Fix:** rsync the artifact tree from NUC and add it to LD_LIBRARY_PATH (Isaac-GR00T INSTALL_FROM_SCRATCH §9-5/§9-6 has the exact rsync + activate.d snippet). Summary:

```bash
rsync -av kist@192.168.1.12:~/fairo/polymetis/polymetis/build/torch_isolation/ \
    ~/fairo-polymetis/polymetis/build/torch_isolation/
rsync -av --include='*.so*' --include='*/' --exclude='*' \
    kist@192.168.1.12:~/miniconda3/envs/polymetis-local/lib/ \
    ~/fairo-polymetis/nuc_libs/
# remove libs that conflict with system curl/SSL
rm -f ~/fairo-polymetis/nuc_libs/libcurl.so* \
      ~/fairo-polymetis/nuc_libs/libssl.so* \
      ~/fairo-polymetis/nuc_libs/libcrypto.so*
# add to LD_LIBRARY_PATH on conda activate
mkdir -p ~/anaconda3/envs/groot-client/etc/conda/activate.d
cat > ~/anaconda3/envs/groot-client/etc/conda/activate.d/polymetis_libs.sh <<'EOF'
export LD_LIBRARY_PATH="${HOME}/fairo-polymetis/polymetis/build/torch_isolation:${HOME}/fairo-polymetis/polymetis/build/torch_isolation/pinocchio_isolation:${HOME}/fairo-polymetis/nuc_libs:${HOME}/anaconda3/envs/groot-client/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
EOF
```

### #7 — `Version mismatch between client and server`

**Cause:** `_version.py` stub on pro4000 doesn't match the NUC's `git describe`.

**Fix:**

```bash
# On NUC
cd ~/fairo && git describe --tags     # e.g. v1.0-1251-g0a01a7fa7
# On pro4000
cat > ~/fairo-polymetis/polymetis/python/polymetis/_version.py <<EOF
__version__ = '1251_g0a01a7fa7'
EOF
```

### #8 — `KeyError: 'CONDA_PREFIX'` on `import polymetis`

**Cause:** Polymetis Python looks up `os.environ["CONDA_PREFIX"]` to find its build artifacts. You imported it without `conda activate groot-client` first.

**Fix:** `conda activate groot-client` (or `conda activate polymetis-local` if on the NUC).

### #9 — `curl: error while loading shared libraries: libcrypto.so.1.1: cannot open shared object file`

**Cause:** `groot-client`'s `LD_LIBRARY_PATH` puts `~/fairo-polymetis/nuc_libs/` first, but we deliberately removed `libcrypto.so.1.1` from there (#6). Ubuntu 22.04's system curl is built against `libcrypto 3.0`, which is also missing from `nuc_libs`. So `curl` searches LD_LIBRARY_PATH first, finds neither, and dies.

**Fix:** unset LD_LIBRARY_PATH for the curl call.

```bash
env -u LD_LIBRARY_PATH curl …
# or
conda deactivate && curl … && conda activate groot-client
```

This affects pyzed wheel download, ZED calibration download, etc.

---

## Cameras

### #10 — ZED 2i UVC missing (HID device only) / `Maybe the USB cable is bad?`

**Cause:** USB 2.0 cable. ZED 2i exposes the UVC interface only on USB 3.0 SuperSpeed (5 Gbps).

**Fix:** Replace cable with a known-good USB 3.0 SuperSpeed C-C. Verify with `lsusb -t | grep 5000M` showing both UVCs at 5 Gbps.

### #11 — ZED open `CALIBRATION FILE NOT AVAILABLE`

**Cause:** Calibration not on disk + ZED's internal libcurl conflicts with the conda LD_LIBRARY_PATH (#9), so it can't download on demand.

**Fix:** Pre-download once (Phase F-2):

```bash
sudo mkdir -p /usr/local/zed/settings
for SN in 33538770 11667817; do
    sudo curl -fsSL -o /usr/local/zed/settings/SN${SN}.conf \
        "https://calib.stereolabs.com/?SN=${SN}"
done
```

If even that download dies, do the `env -u LD_LIBRARY_PATH curl …` workaround from #9.

### #12 — ZED Mini disappears after 2 hot-plugs

**Cause:** Linux uvcvideo driver + ZED Mini sometimes wedge in a half-enumerated state.

**Fix:** Unplug, wait 5 s, plug back into a *different* USB port. (The same port often comes back in the wedged state until a full PC reboot.) Future improvement: udev rule that resets the device on disconnect — not yet written.

---

## ART gripper (pro4000)

### #13 — `art-gripper-daemon` enters `failed` state after fault

**Symptom:** `systemctl is-active art-gripper-daemon` prints `failed`. Logs (`journalctl -u art-gripper-daemon -n 50`) show `Slave has not entered OP state` or similar.

**Fix:**

```bash
sudo bash ~/Hyundai_motors_Gripper/scripts/restart_gripper.sh
```

Stops the daemon → reloads `ec_master` and `ec_generic` kernel modules → restarts the daemon → pings the new TCP `:50053` to verify. If the gripper is still latched after this, 24 V power-cycle the gripper itself; the daemon auto-reattaches on the next EtherCAT scan.

### #14 — Gripper doesn't actuate on Vive Trigger after a HOME

**Cause (early):** ART's `last_teleop_cmd` was stale (still CLOSE from before HOME); the first non-NONE command after HOME was the same value as the last seen, so the equality check skipped it.

**Fix:** the ART controller in this repo resets `last_teleop_cmd` to `NONE` whenever the incoming command is `NONE`, and the explicit GOTO at HOME also writes `last_teleop_cmd = NONE`. Plus `vive_teleop_process.py` resets the toggle state machine (`gripper_closed`, `awaiting_trigger_release`) at the end of HOME. With both fixes the very next Trigger press after HOME actuates correctly.

---

## OpenCV / cv2 viewer

### #15 — `c` key inside the cv2 window doesn't start recording

**Cause (early):** `cv2_viewer.py` only forwarded `q` (SIGINT). Other keys never reached the demo.

**Fix:** the viewer in this repo forwards `c`→SIGUSR1 (record start), `s`→SIGUSR2 (record stop), `h`→SIGHUP (HOME), `q`→SIGINT (quit). The demo (`scripts_real/demo_franka_vive.py`) registers handlers and translates them into env method calls.

### #16 — `??? IDLE ??? RECORDING` garbled banners in cv2 window

**Cause:** em-dash (`—`) in the `cv2.putText` strings; cv2's bundled font has no Unicode glyph table.

**Fix:** all banner strings in `demo_franka_vive.py` use ASCII colons (`:`) only. Don't reintroduce em-dashes, en-dashes, or smart quotes.

### #17 — cv2 window never appears + trackpad-HOME polling halts

**Cause:** the controlling shell sent SIGTSTP (Ctrl-Z hit by accident) → whole foreground process group stopped. Child Vive/controller processes keep running, but the main loop is frozen.

**Fix:** use `bin/run_test_session.sh` which `setsid`s the demo into its own session. Job-control signals from the launching shell can't reach it.

### #18 — `q` in cv2 window quits the viewer but demo subprocesses linger

**Symptom:** `pgrep -af demo_franka_vive` shows 6+ orphaned children for several seconds after the cv2 window closes.

**Cause:** `multiprocessing.Process.terminate()` is asynchronous — Python returns from the join before the OS has actually reaped the children.

**Mitigation:** `run_test_session.sh` waits 2 seconds, counts orphans, and prints a `WARNING:` line if any are still alive, with the `pkill -KILL -f demo_franka_vive` command to force-clean. In practice the count is 0 by the time the warning would have fired; the line exists only as a safety net.

---

## SteamVR / Vive

### #19 — `vrserver --keepalive` exits within 30 s

**Cause:** SteamVR's first-run wizard hasn't been completed. `--keepalive` doesn't bypass first-run.

**Fix:** Once, in a graphical session, launch SteamVR through the Steam GUI. Pair the controller. Run through "Setup Lighthouses". After it succeeds, `vrserver --keepalive` works headlessly forever.

### #20 — `lsusb | grep -i htc` shows nothing

**Cause:** Vive base stations not powered, or controller not paired to your local SteamVR install.

**Fix:** Check the green LEDs on both Lighthouses. Re-pair the controller through SteamVR (Steam GUI → SteamVR → Devices → Pair Controller).

---

## Franka Desk

### #21 — FCI Activate is greyed out / "End Effector: (none)"

**Cause:** EE connector not seated, or wrong EE selected.

**Fix:** physically re-seat the connector at the flange, power-cycle the Franka Controller Box, log into Desk, set EE to "Franka Hand" or "Custom" (with ART), Save, then Activate FCI.

### #22 — `franka::CommandException libfranka gripper: Command failed!` SIGABRT

**Cause:** Franka Hand metadata mismatch on the controller box (firmware thinks max_width=13 mm). Doesn't affect ART workflow.

**Fix:** Franka Desk → Settings → End Effector → re-select Franka Hand → Save. If that doesn't take, full power-cycle the controller box.

---

## Misc

### #23 — `pkill -f` accidentally kills your shell

**Cause:** the search pattern matches the bash command line that's running `pkill`, so you reap your own shell.

**Fix:** prefer `pgrep -f … | xargs -r kill` — `pgrep` doesn't see itself in the result list.

### #24 — `tail --pid=$DEMO_PID` hangs after demo dies

This is intentional behavior of `tail` — the `--pid` option polls. But the
launcher (`run_test_session.sh`) does this on purpose: the user can Ctrl-C
the tail to detach without affecting the demo, and the tail self-exits when
the demo dies, freeing the shell.

If you don't want this behavior, run `bash bin/start_teleop.sh …` directly
without the launcher.

---

## Stabilization catalog (2026-05-09 KIST teleop session)

### #25 — Reflex cascade after recovery (8-13 reflex storms in 1.5 min window)

**Symptom**: After the first `communication_constraints_violation` reflex,
recoveries fire every ~3 s for 1-2 min before settling. User experience:
robot stuck for tens of seconds even though each individual recovery is fast.

**Root cause (multiple stacked)**:
1. Pro4000 NIC IRQ for the NUC subnet was floating across CPUs 3 + 12
   (default IRQ balance). NIC softirq competes with Python compute on the
   same cores → gRPC `update_desired_joint_positions` gets delayed →
   libfranka 1 ms FCI window missed → reflex.
2. Pro4000's `FrankaInterpolationController` ran on whatever core the
   default scheduler picked. With ~8 Python multiprocessing children on a
   14-core CPU, contention is statistical: usually fine, but bursts of
   activity (gripper transition, ZED frame burst) push the controller off
   schedule.
3. `time.sleep(1.5)` in the recovery handler did NOT reset the loop's
   `t_start` / `iter_idx` clock. After the sleep, `t_wait_util = t_start +
   iter_idx*dt` was 1.5 s in the past → `precise_wait` returned immediately
   → 100-150 iters fired at 250 Hz to "catch up" → polymetis HybridJoint
   ImpedanceController got hammered with updates faster than it could
   process → another reflex.

**Fix**:
- `install/pro4000/sbin/franka_client_rt_apply.sh` pins enp130s0 NIC IRQ
  to cores 0,1, sets governor=performance, stops irqbalance.
- `polymetis_franka_teleop/common/realtime_util.py` pins
  `FrankaInterpolationController` to cores 6,7 + SCHED_RR/20, and
  `ArtGripperController` to cores 8,9 + SCHED_RR/15.
- `franka_vive_env.py` start() pins the main demo + general children to
  cores 0-5 + 10-13 (away from RT cores 6-9).
- `franka_interpolation_controller.py` recovery path resets `t_start` /
  `iter_idx` after the impedance restart, eliminating the catch-up burst.
- /etc/security/limits.d/franka_client_rt.conf grants rtprio=50 / nice=-15;
  /etc/systemd/system.conf.d/franka_rt.conf sets DefaultLimitRTPRIO=50 so
  the user's desktop session also gets the bump (was being capped at 0 by
  systemd's user@.service slice default).

**After fix**: 11 → 0 NUC-side reflexes per session at KIST.

---

### #26 — Stuck recovery: "Grip" not responding after reflex; only trackpad-HOME escapes

**Symptom**: After a reflex+recovery, user re-engages Vive Grip and sees
"Clutch ENGAGED", but the robot does not move. Pressing trackpad HOME makes
the robot move to ready pose, and from there normal teleop resumes.

**Root cause**: The `FrankaInterface` IK silent-failure path:

```python
joint_target, success = self.robot.solve_inverse_kinematics(...)
if not success:
    if self._last_good_joint_target is not None:
        self.robot.update_desired_joint_positions(self._last_good_joint_target)
    return
```

After a reflex, the robot is at a pose near the reflex configuration. The
IK seed (`_cached_q`) and last-good fallback (`_last_good_joint_target`)
both still hold the values from BEFORE the reflex. When the user moves the
Vive controller toward a target close to the previous reflex configuration,
IK seeds from the bad pose, fails, and the fallback path replays the same
bad joint target every tick. Robot stays put even though Grip is engaged.

`move_to_joint_positions` (the HOME path) does NOT use IK — it uses a
trajectory tracking controller. So HOME works regardless of stuck IK. After
HOME, both `_cached_q` and `_last_good_joint_target` are naturally
overwritten with fresh values, and IK works again.

**Fix**: `FrankaInterface.reset_ik_state()` is called explicitly on every
recovery and after every HOME (`franka_interpolation_controller.py:reset_
ik_state`). Forces the next IK call to re-fetch joint positions from the
robot, breaking the stale-cache lock.

---

### #27 — `start_cartesian_impedance` silently fails after consecutive reflexes; user gets stuck

**Symptom**: After 2-3 reflex+recovery cycles in quick succession,
`start_cartesian_impedance(Kx, Kxd)` returns success but polymetis returns
`Unable to update desired joint positions. Use 'start_joint_impedance' to
start a joint impedance controller.` for every subsequent
`update_desired_joint_positions` call. The recovery handler keeps retrying
forever.

**Root cause**: Polymetis's HybridJointImpedanceController state machine
appears to enter a confused state when `start_cartesian_impedance` is
called multiple times within a short window without a clean transition.
Only `move_to_joint_positions` (a different polymetis controller path)
forces a clean state reset that subsequent `start_cartesian_impedance`
calls can actually take.

**Fix**: Auto-HOME escalation in `franka_interpolation_controller.py`. The
recovery handler tracks `consecutive_recovery_count`. If 2 recoveries fire
within a 10 s window, the next iteration injects a synthetic
`Command.MOVE_HOME` into the input queue. HOME forces the polymetis state
reset, and `start_cartesian_impedance` works correctly afterward. User does
not need to press trackpad HOME manually.

Log signature when this fires:
```
[FrankaPositionalController] !! 2 recoveries in <10 s -- start_cartesian_
impedance not taking. Auto-HOME to force polymetis state reset.
[FrankaPositionalController] auto-HOME injected to force polymetis state
reset (catalog #27)
```

---

### #28 — Joint near-singularity reflexes (j3=2.97, j5=2.97, j6=2.89)

**Symptom**: Robot reflex fires when user pushes the elbow (j3) or wrist
(j5/j6) toward the Franka mechanical limit. NUC log shows
`Joint velocity violation` or `joint_position_limits` alongside the
`communication_constraints_violation`.

**Root cause**: At joint angles within ~0.3 rad of the mechanical limit,
Cartesian impedance requires high joint torque to maintain the EEF target.
libfranka's `safety_controller` activates at lower torque thresholds in
this regime, and small subsequent motion trips the limit.

**Fix**: Joint-limit early-warning in `franka_interpolation_controller.py`
at every iter:
- j3 / j5 warn at |q| > 2.5 rad (margin 0.47 rad)
- j6 warns at |q| > 2.5 rad (margin 0.39 rad)
- Throttled to 1 message per joint per 10 s

User-side mitigation: back off the corresponding joint angle. Most often
the elbow (j3) — fold the arm back closer to the base before pushing the
EEF further out.

The cv2 status panel (`scripts_real/demo_franka_vive.py:644-672`) also
shows live j3/j5/j6 magnitude and margin in green/amber/red color so the
user can avoid the limit before reflex fires.

---

### #29 — Controller-not-ready transient race after `start_cartesian_impedance`

**Symptom**: Even with no reflex on the NUC side, pro4000 sometimes sees
`Tried to perform a controller update with no controller running` for a
few iterations after `start_cartesian_impedance` returns.

**Root cause**: `start_cartesian_impedance` returns success as soon as
polymetis QUEUES the policy load, but the controller takes 50-400 ms more
to actually accept commands. During that window any update is rejected.

**Fix**: `FrankaInterface.wait_until_controller_ready()` polls
`update_desired_joint_positions(current_q)` (a no-op feed since target ==
current) until it stops raising. Replaces the previous
`time.sleep(1.5)` blind wait. Typical latency 100-300 ms; 1.5 s timeout
matches the previous behavior. Reduces both: false-recovery races and
user-perceived recovery duration.

---

### #30 — Gripper close (`grasp(timeout_s=2.0)`) blocks the gripper loop, causes 60-100% overruns

**Symptom**: Every time the user presses the Vive Trigger to close the
ART gripper, the next 5 s of `ArtGripperController.run()` reports 100-200
overruns/300 (33-67%) — the loop falls behind by ~2 s.

**Root cause**: `art_gripper_client.ArtGripperInterface.grasp(timeout_s=
2.0)` is a *blocking* call: it waits up to 2 s for the firmware to report
"idle" before returning. The 60 Hz state-polling loop is single-threaded,
so 2 s of blocking == 120 missed iterations.

**Fix**: `art_gripper_controller.py` teleop close path now uses
`gripper.goto(width=close_w, blocking=False)` instead of `grasp()`. The
EtherCAT slave still completes the trajectory and caps fingertip force via
the current limit, but the controller loop returns immediately. Force-mode
grasp (i.e., closing until a force threshold is met) is still available
via the explicit `Command.GRASP` input-queue command path.

---

### #31 — Stale Python child holds /dev/video0 after demo crash → next demo sees only 1 ZED

**Symptom**: After a demo crash mid-init (e.g., the AttributeError race in
catalog #29), restarting the demo shows `pyzed.sl.Camera.get_device_list()`
returning only 1 camera even though `lsusb` shows both ZEDs.

**Root cause**: The ZED SDK opens `/dev/videoN` exclusively. A leftover
multiprocessing child from the previous failed demo still holds the file
descriptor, blocking the new demo's open call.

**Fix**: `bin/preflight_full.sh` step [1/6] kills any process matching
`demo_franka_vive | spawn_main | cv2_viewer | FrankaPositional |
ArtGripper | SingleZed | MultiZed | ViveTeleop` before starting. Step
[3/6] then verifies pyzed sees both cameras; if not, falls through to a
warning telling the user which lsof PID is holding the device.

---

### #32 — ART daemon TCP single-client lock held by killed previous client

**Symptom**: ART daemon is `systemctl is-active` but every
`ArtGripperInterface(...)` call times out. `ss -tn` shows several
CLOSE-WAIT / FIN-WAIT-2 / ESTAB connections to `:50053`.

**Root cause**: The ART daemon (`Hyundai_motors_Gripper/src/server.cpp`)
serves one TCP client at a time. When a previous client process is
SIGKILL'd while connected, the daemon's accept-loop is still blocked on
that dead socket; new connects hang.

**Fix**:
- `bin/preflight_full.sh` step [2/6] sends an actual OP_PING (binary
  protocol) with a 3 s timeout. If it times out, runs
  `~/Hyundai_motors_Gripper/scripts/restart_gripper.sh` (which stops the
  daemon, reloads the EtherCAT kmod, restarts the daemon, and re-pings).
- The daemon-side fix would be to detect dead connections and kick them;
  this is outside our repo, see Hyundai_motors_Gripper.

---

### #33 — systemd-logind caps desktop-terminal `ulimit -r` at 0 even with PAM `limits.conf`

**Symptom**: `/etc/security/limits.d/franka_client_rt.conf` sets `kist - rtprio 50` but a freshly opened desktop terminal (gnome-terminal /
ctrl+alt+t) reports `ulimit -r = 0`. SSH-to-localhost reports 50 correctly.

**Root cause**: GDM/gnome-session creates a `user@1000.service` slice via
systemd. systemd applies its own `DefaultLimitRTPRIO=0` (compile-time
default) to processes inside the slice AFTER pam_limits.so. The desktop
terminal inherits the slice's limit, ignoring PAM. SSH bypasses the slice
because sshd's own session is launched outside `user@*.service`.

**Fix**: drop-in `/etc/systemd/system.conf.d/franka_rt.conf` and
`/etc/systemd/user.conf.d/franka_rt.conf` set `DefaultLimitRTPRIO=50`,
`DefaultLimitNICE=-15`, `DefaultLimitMEMLOCK=infinity`. Takes effect on
the next desktop logout/login. Both files installed by
`install/install_pro4000_rt.sh`.
