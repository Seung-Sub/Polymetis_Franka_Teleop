#!/usr/bin/env python3
"""Diffusion-Policy inference daemon.

Runs in the ``umi`` conda env so torch 2.11+cu128 can drive the RTX PRO 4000
Blackwell (sm_120) GPU. The Polymetis-bound main eval loop runs in a separate
``groot-client`` env (CPU torch 1.13) and forwards observations via a ZMQ
REQ/REP socket on localhost.

Protocol
--------
The daemon waits on ``tcp://127.0.0.1:<port>`` for a single REQ payload (msgpack):

    request = {
        'op':    'predict',
        'obs':   {key: np.ndarray, ...},     # shape: (B=1, T_obs, ...)
    }

and replies with:

    reply = {
        'ok':           bool,
        'action':       np.ndarray shape (B, n_action_steps, action_dim),
        'inference_ms': float,
        'error':        str,                 # only when ok=False
    }

Special ops:
  * 'ping'      -> {'ok': True, 'pong': True, 'inference_ms': 0.0}
  * 'shutdown'  -> {'ok': True}; daemon exits.

Usage:
  python scripts_real/eval_dp_inference_daemon.py \\
      --ckpt checkpoints/dp/unet_epoch100.ckpt \\
      --port 5555 \\
      --device cuda:0
"""
import os, sys, time, traceback
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

_DP_PATH = os.environ.get('DIFFUSION_POLICY_PATH',
                          os.path.expanduser('~/diffusion_policy'))
if os.path.isdir(_DP_PATH) and _DP_PATH not in sys.path:
    sys.path.insert(0, _DP_PATH)

import click
import numpy as np
import torch
import dill
import hydra
import zmq
import msgpack
import msgpack_numpy as mnp
mnp.patch()

from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

# Calibration-input collection lives in THIS process: the diffusion UNet
# (diffusion_unet_hybrid_image_policy.conditional_sample) calls
# globalvar.appendInput((trajectory, t, global_cond)) on every sampling step,
# and that runs inside policy.predict_action() right here in the daemon. The
# eval client (collect_calib_data.py) is a SEPARATE process, so it can never
# read this list — the daemon must be the one that saves it. See 'save_calib'.
import quant.calib.dp_globalvar as globalvar

OmegaConf.register_new_resolver("eval", eval, replace=True)


@click.command()
@click.option('--ckpt', required=True, help='Path to DP workspace ckpt.')
@click.option('--port', default=5555, type=int)
@click.option('--device', default='cuda:0')
@click.option('--num_inference_steps', default=8, type=int)
@click.option('--use_ema/--no_ema', default=True)
@click.option('--clip_sample_mode',
              type=click.Choice(['auto', 'force_off', 'keep']),
              default='auto',
              help="DDIM noise_scheduler.clip_sample handling. "
                   "'auto' (default): inspect ckpt's action normalizer — if it is "
                   "identity (scale=1, offset=0) the model was trained with the "
                   "old RobomimicReplayImageDataset abs_action=False branch and "
                   "MUST be evaluated with clip_sample=False (otherwise the "
                   "training distribution aa[0]∈[+2.05,+3.40] is unreachable). "
                   "If the action normalizer is a true range/min-max normalizer "
                   "(force_action_range_normalize=True training), clip_sample is "
                   "kept at the training-time value. 'force_off' always sets to "
                   "False (legacy fallback). 'keep' never touches it.")
def main(ckpt, port, device, num_inference_steps, use_ema, clip_sample_mode):
    print(f"[daemon] loading {ckpt}")
    payload = torch.load(ckpt, pickle_module=dill, map_location='cpu', weights_only=False)
    cfg = payload['cfg']
    print(f"[daemon] workspace: {cfg._target_}")
    cls = hydra.utils.get_class(cfg._target_)
    ws: BaseWorkspace = cls(cfg)
    ws.load_payload(payload, exclude_keys=None, include_keys=None)
    policy: BaseImagePolicy = ws.ema_model if (use_ema and cfg.training.use_ema) else ws.model
    policy.eval().to(device)
    policy.num_inference_steps = num_inference_steps

    # === Action normalizer auto-detection (2026-05-21) ===
    # Old ckpts (RobomimicReplayImageDataset abs_action=False default) ship
    # an IDENTITY action normalizer (scale=1, offset=0) and need
    # clip_sample=False at inference. New ckpts trained with
    # force_action_range_normalize=True have a true range normalizer
    # (scale != 1) and can use the DDIM-default clip_sample=True.
    sds = ws.ema_model.state_dict() if (use_ema and cfg.training.use_ema) else ws.model.state_dict()
    act_scale = sds.get('normalizer.params_dict.action.scale', None)
    is_identity_normalizer = (
        act_scale is not None and
        torch.allclose(act_scale, torch.ones_like(act_scale), atol=1e-6) and
        torch.allclose(sds['normalizer.params_dict.action.offset'],
                       torch.zeros_like(sds['normalizer.params_dict.action.offset']),
                       atol=1e-6)
    )
    print(f"[daemon] action normalizer = "
          f"{'IDENTITY (legacy ckpt)' if is_identity_normalizer else 'range/min-max (modern ckpt)'}")

    if clip_sample_mode == 'force_off':
        do_disable_clip = True
        reason = "explicit --clip_sample_mode=force_off"
    elif clip_sample_mode == 'keep':
        do_disable_clip = False
        reason = "explicit --clip_sample_mode=keep"
    else:  # auto
        do_disable_clip = is_identity_normalizer
        reason = ("auto: identity normalizer detected, disabling clip_sample"
                  if do_disable_clip else
                  "auto: range normalizer detected, keeping clip_sample at training default")

    if do_disable_clip:
        prev = policy.noise_scheduler.config.clip_sample
        policy.noise_scheduler.config.clip_sample = False
        print(f"[daemon] noise_scheduler.clip_sample {prev} -> False ({reason})")
    else:
        print(f"[daemon] noise_scheduler.clip_sample kept at "
              f"{policy.noise_scheduler.config.clip_sample} ({reason})")

    print(f"[daemon] policy on {device}; horizon={cfg.horizon} n_obs_steps={cfg.n_obs_steps}")

    # warm up
    sample_obs = {
        'exterior_image_1_left': torch.zeros(1, cfg.n_obs_steps, 3, 224, 224, device=device),
        'wrist_image_left':      torch.zeros(1, cfg.n_obs_steps, 3, 224, 224, device=device),
        'robot0_eef_pos':        torch.zeros(1, cfg.n_obs_steps, 3, device=device),
        'robot0_eef_quat':       torch.tensor([[[0,0,0,1]]*cfg.n_obs_steps], dtype=torch.float32, device=device),
        'robot0_eef_rot_axis_angle': torch.zeros(1, cfg.n_obs_steps, 3, device=device),
        'robot0_gripper_qpos':   torch.ones(1, cfg.n_obs_steps, 1, device=device),
    }
    for _ in range(2):
        with torch.no_grad():
            _ = policy.predict_action(sample_obs)
    print(f"[daemon] warmed up; listening on tcp://127.0.0.1:{port}")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{port}")

    n_handled = 0
    print(f"[daemon] entering recv loop at {time.time():.3f}", flush=True)
    try:
        while True:
            # Verbose first-3 recvs so a 'daemon never recv'd' failure leaves
            # evidence. 2026-05-21 01:18 run showed eval client.predict timing
            # out at 10s while daemon log stopped at 'listening' -- couldn't
            # tell if recv() blocked or if msgpack/torch never returned.
            if n_handled < 3:
                t_pre = time.time()
                print(f"[daemon] sock.recv() wait (n_handled={n_handled})...", flush=True)
                raw = sock.recv()
                print(f"[daemon] sock.recv() returned {len(raw)} bytes after "
                      f"{(time.time()-t_pre)*1000:.1f}ms", flush=True)
            else:
                raw = sock.recv()
            req = msgpack.unpackb(raw, raw=False)
            op = req.get('op', 'predict')
            if n_handled < 3:
                print(f"[daemon] op={op}", flush=True)

            if op == 'ping':
                sock.send(msgpack.packb({'ok': True, 'pong': True, 'inference_ms': 0.0}))
                continue
            if op == 'save_calib':
                # The calibration inputs accumulated in THIS process via
                # globalvar.appendInput during every predict. The client asks
                # us to persist them (it cannot — its own globalvar is empty).
                try:
                    out_path = req['path']
                    def _to_cpu(x):
                        if torch.is_tensor(x):
                            return x.detach().cpu()
                        if isinstance(x, (tuple, list)):
                            return type(x)(_to_cpu(v) for v in x)
                        return x
                    data = [_to_cpu(item) for item in globalvar.getInputList()]
                    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
                    torch.save(data, out_path)
                    print(f"[daemon] saved {len(data)} calib inputs -> {out_path}",
                          flush=True)
                    globalvar.resetInput()
                    sock.send(msgpack.packb({'ok': True, 'n': len(data), 'path': out_path}))
                except Exception as e:
                    tb = traceback.format_exc()
                    sock.send(msgpack.packb({'ok': False, 'error': f"{type(e).__name__}: {e}\n{tb}"}))
                continue
            if op == 'shutdown':
                sock.send(msgpack.packb({'ok': True}))
                print(f"[daemon] shutdown after {n_handled} predicts")
                break

            try:
                obs_np = req['obs']
                obs = {}
                for k, v in obs_np.items():
                    obs[k] = torch.as_tensor(v, device=device)
                t0 = time.time()
                with torch.no_grad():
                    out = policy.predict_action(obs)
                t_inf = time.time() - t0
                action = out['action'].detach().cpu().numpy()
                reply = {'ok': True, 'action': action, 'inference_ms': t_inf * 1000.0}
                # FIX A (receding-horizon overlap): also ship the FULL predicted
                # horizon (action_pred, length=cfg.horizon=16) plus the policy's
                # n_obs_steps. action_pred[k] is the action at trajectory step k,
                # and steps 0..n_obs_steps-1 align with the observation window so
                # action_pred[n_obs_steps-1] lands at the last obs time. The eval
                # loop uses this to schedule >1 inference-period of future
                # waypoints, so consecutive inference chunks overlap and the arm
                # never reaches the end of a chunk and stalls before the next
                # (the 200ms hold-and-jump that made deploy jerky). 'action'
                # (n_action_steps=8) is kept unchanged for backward compatibility
                # (collect_calib_data.py simply ignores the extra keys).
                if 'action_pred' in out:
                    ap_np = out['action_pred'].detach().cpu().numpy()
                    reply['action_pred'] = ap_np
                    reply['n_obs_steps'] = int(cfg.n_obs_steps)
                    # Warn ONLY when the RAW policy chunk is degenerate: the
                    # identity-normalizer ckpts run with clip_sample=False, so an
                    # OOD obs can yield an out-of-range, discontinuous chunk
                    # (max consecutive step >> the normal <=~45mm; grip outside
                    # [0,1]). The eval client holds these (DEGENERATE_INTRA_STEP_M);
                    # this line makes them visible/countable without per-inference
                    # spam. A high rate of these = the policy is seeing OOD states
                    # (retrain with a range normalizer to let clip_sample clamp).
                    _ap = ap_np[0]
                    _dp = (np.linalg.norm(np.diff(_ap[:, :3], axis=0), axis=1) * 1000.0
                           if len(_ap) > 1 else np.array([0.0]))
                    if _dp.max() > 100.0:
                        _aa = np.linalg.norm(_ap[:, 3:6], axis=1)
                        print(f"[daemon/diag] DEGENERATE raw chunk: max_pos_step={_dp.max():.0f}mm "
                              f"mean={_dp.mean():.0f}mm  aa_norm=[{_aa.min():.3f},{_aa.max():.3f}] "
                              f"grip=[{_ap[:,6].min():.2f},{_ap[:,6].max():.2f}]", flush=True)
                n_handled += 1
                if n_handled % 10 == 0:
                    print(f"[daemon] handled {n_handled} predicts, last inference={t_inf*1000:.1f}ms")
                sock.send(msgpack.packb(reply))
            except Exception as e:
                tb = traceback.format_exc()
                sock.send(msgpack.packb({'ok': False, 'error': f"{type(e).__name__}: {e}\n{tb}"}))
    finally:
        sock.close()
        ctx.term()
        print("[daemon] exit")


if __name__ == '__main__':
    main()
