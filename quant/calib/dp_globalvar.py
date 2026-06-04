"""
To collect input data for naive quantization's calibration data.

The diffusion UNet/DiT conditional_sample calls appendInput() on every denoising
step. That is only wanted during CALIBRATION COLLECTION -- during normal FP /
quantized deploy it would accumulate (trajectory, global_cond) GPU tensors every
inference and never free them (the daemon only resetInput()s inside save_calib).
So appendInput is GATED on `enabled` (default False). The daemon flips it on with
--collect_calib (set by bin/run_eval_dp_2.sh); plain deploy leaves it off, so the
hook is a single cheap bool check with zero accumulation.
"""

# collect input data for calibration data
global diffusion_input_list
diffusion_input_list = []

# Gate: only accumulate when calibration collection is explicitly enabled.
enabled = False

def set_enabled(flag=True):
    global enabled
    enabled = bool(flag)

def appendInput(value):
    if not enabled:
        return
    diffusion_input_list.append(value)

def getInputList():
    return diffusion_input_list

def resetInput():
    diffusion_input_list.clear()