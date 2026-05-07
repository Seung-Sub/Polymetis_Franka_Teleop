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
