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


# ---------------------------------------------------------------------------
# Quantized / distilled checkpoint support (PTQ naiveq + QALoRA/TaDA).
# Mirrors agile_diffusion/quant/eval/eval_fakeq.py::build_student_from_policy so
# the SAME quantw{b}a{b}_naiveq.pth / qalora state_dicts deploy here unchanged.
# All quant imports are lazy (inside build_quant_model) so the FP/agile-DP path
# never needs the quant package.
# ---------------------------------------------------------------------------
import inspect as _inspect


class _DPNoisePredWrapper(torch.nn.Module):
    """Adapt the DP noise-pred UNet to QuantModel's (x, t, global_cond) calls."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.sig = _inspect.signature(net.forward)

    def forward(self, x, t, global_cond=None, **kwargs):
        dev = x.device
        t = (t.to(device=dev, dtype=torch.long) if torch.is_tensor(t)
             else torch.as_tensor(t, device=dev, dtype=torch.long))
        if global_cond is not None:
            if not torch.is_tensor(global_cond):
                global_cond = torch.as_tensor(global_cond)
            global_cond = global_cond.to(device=dev, dtype=x.dtype)
        if 'global_cond' in self.sig.parameters:
            return self.net(x, t, global_cond=global_cond, **kwargs)
        if 'cond' in self.sig.parameters:
            return self.net(x, t, cond=global_cond, **kwargs)
        return self.net(x, t)


def _get_calib_samples(loader, n):
    tr, tt, cc = [], [], []
    for (traj, t, cond) in loader:
        tr.append(traj); tt.append(t); cc.append(cond)
        if len(tr) >= n:
            break
    return torch.cat(tr, 0)[:n], torch.cat(tt, 0)[:n], torch.cat(cc, 0)[:n]


def _load_quant_state_dict(qnn, qckpt):
    # key remap identical to eval_fakeq.load_state_dict_safely: policy.state_dict()
    # nests the quant net under 'model.model....' (policy.model == qnn, qnn.model
    # == wrapped UNet), so strip one 'model.' so it lines up with qnn's own keys.
    new = {}
    for k, v in qckpt.items():
        nk = k
        if nk.startswith('model.model.'):
            nk = nk.replace('model.model.', 'model.', 1)
        elif nk.startswith('net.'):
            nk = 'model.net.' + nk[len('net.'):]
        new[nk] = v
    missing, unexpected = qnn.load_state_dict(new, strict=False)
    print(f"[daemon/quant] load_state_dict: missing={len(missing)} "
          f"unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"[daemon/quant]   missing[:8]={missing[:8]}", flush=True)
    return missing, unexpected


def build_quant_model(policy, quant_ckpt, qtype, n_bit, calib_data, num_steps, device):
    """Replace policy.model with a quantized UNet loaded from quant_ckpt.

    The init pass (need_init=True forward on calib samples) materialises the
    quant params' tensors; the saved state dict then overwrites their values.
    """
    from torch.utils.data import DataLoader
    from quant.quant_model import QuantModel, QuantModelLoRA, QuantModelLoRATaDA
    from quant.quant_dataset import DiffusionInputDataset
    base = _DPNoisePredWrapper(policy.model).eval().to(device)
    wq = {'n_bits': n_bit, 'channel_wise': True,  'scale_method': 'max', 'symmetric': True}
    aq = {'n_bits': n_bit, 'channel_wise': False, 'scale_method': 'max',
          'leaf_param': True, 'symmetric': True}
    if qtype == 'naiveq':
        qnn = QuantModel(model=base, weight_quant_params=wq, act_quant_params=aq, need_init=True)
    elif qtype == 'qalora':
        qnn = QuantModelLoRA(model=base, weight_quant_params=wq, act_quant_params=aq, num_steps=num_steps)
    elif qtype == 'qalora_tada':
        qnn = QuantModelLoRATaDA(model=base, weight_quant_params=wq, act_quant_params=aq, num_steps=num_steps)
    else:
        raise ValueError(f"unknown qtype {qtype!r} (expected naiveq|qalora|qalora_tada)")
    qnn.set_first_last_layer_to_8bit()
    qnn.set_quant_state(True, True)
    qnn = qnn.to(device).eval()
    ds = DiffusionInputDataset(calib_data)
    dl = DataLoader(dataset=ds, batch_size=16, shuffle=True)
    ct, cti, cc = _get_calib_samples(dl, 4000)
    print(f"[daemon/quant] init pass on {ct.shape[0]} calib samples ({qtype} "
          f"w{n_bit}a{n_bit}) ...", flush=True)
    with torch.no_grad():
        _ = qnn(ct.to(device), cti.to(device), cc.to(device))
    qckpt = torch.load(quant_ckpt, map_location='cpu')
    _load_quant_state_dict(qnn, qckpt)
    qnn.eval()
    setattr(policy, 'model', qnn)
    print(f"[daemon/quant] loaded {qtype} w{n_bit}a{n_bit} from {quant_ckpt}", flush=True)
    return policy


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
@click.option('--quant_ckpt', default=None,
              help='Optional quantized state_dict (quantw{b}a{b}_naiveq.pth or '
                   'a qalora .pth). When set, --ckpt is the FP teacher and its '
                   '.model is replaced by the quantized UNet. Omit for FP/agile-DP.')
@click.option('--qtype', type=click.Choice(['naiveq', 'qalora', 'qalora_tada']),
              default='naiveq', help='Quant model family for --quant_ckpt.')
@click.option('--n_bit', default=8, type=int, help='Weight/act bit-width for --quant_ckpt.')
@click.option('--calib_data', default=None,
              help='DiffusionInput_calib.pth used for the quant init pass '
                   '(required with --quant_ckpt).')
@click.option('--qalora_num_steps', default=100, type=int,
              help='num_steps for QALoRA/TaDA models (ignored for naiveq).')
def main(ckpt, port, device, num_inference_steps, use_ema, clip_sample_mode,
         quant_ckpt, qtype, n_bit, calib_data, qalora_num_steps):
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

    # === Optional: swap in a quantized / distilled UNet ===
    # --ckpt stays the FP teacher (for cfg/normalizer/scheduler); only policy.model
    # is replaced. agile-DP FP ckpts deploy with NO --quant_ckpt (this block is
    # skipped). The action_normalizer/clip_sample logic above is unaffected
    # (it reads the normalizer, which the quant swap does not touch).
    if quant_ckpt:
        if not calib_data:
            raise click.UsageError("--quant_ckpt requires --calib_data for the init pass")
        policy = build_quant_model(
            policy, quant_ckpt=quant_ckpt, qtype=qtype, n_bit=n_bit,
            calib_data=calib_data, num_steps=qalora_num_steps, device=device)
        policy.eval().to(device)
        policy.num_inference_steps = num_inference_steps

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
