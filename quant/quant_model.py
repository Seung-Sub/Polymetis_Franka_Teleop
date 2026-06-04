import torch
import torch.nn as nn
import weakref
from quant.quant_layer import (
    QuantModule,
    QuantModuleLoRA,
    QuantModuleLoRATaDA,
    QuantModuleLoRATaDA_org,
    QuantMultiheadAttention,
    StraightThrough,
    UniformAffineQuantizer,
)

SPECIAL_8BIT_MODULE_KEYS = (
    "net.diffusion_step_encoder.1",
    "net.diffusion_step_encoder.3",
    "net.down_modules.0.0.blocks.0.block.0",
    "net.final_conv.1",
    "net.input_emb",
    "net.cond_obs_emb",
    "net.head",
)


def _get_quantized_module_n_bits(module: nn.Module):
    if hasattr(module, 'weight_quantizer'):
        return getattr(module.weight_quantizer, 'n_bits', None)
    if hasattr(module, 'in_proj_weight_quantizer'):
        return getattr(module.in_proj_weight_quantizer, 'n_bits', None)
    if hasattr(module, 'q_proj_weight_quantizer'):
        return getattr(module.q_proj_weight_quantizer, 'n_bits', None)
    return None


def _set_quantized_module_to_8bit(module: nn.Module):
    if hasattr(module, 'bitwidth_refactor'):
        module.bitwidth_refactor(8)
    else:
        module.weight_quantizer.bitwidth_refactor(8)
        if getattr(module, 'act_quantizer', None) is not None:
            module.act_quantizer.bitwidth_refactor(8)

    if hasattr(module, 'ignore_reconstruction'):
        module.ignore_reconstruction = True


def _apply_first_last_8bit_overrides(root_module: nn.Module, quant_types, verbose: bool = False):
    name2mod = {n: m for n, m in root_module.named_modules() if isinstance(m, quant_types)}
    if not name2mod:
        return

    preferred_keys = list(SPECIAL_8BIT_MODULE_KEYS)

    seen = set()
    found_preferred = False
    for key in preferred_keys:
        if key in name2mod and key not in seen:
            found_preferred = True
            n_bits = _get_quantized_module_n_bits(name2mod[key])
            if (n_bits is not None) and (n_bits >= 8):
                seen.add(key)
                continue
            if verbose:
                print("[INFO] set 8bit for", key)
            _set_quantized_module_to_8bit(name2mod[key])
            seen.add(key)

    if found_preferred:
        return

    ordered = list(name2mod.items())
    fallback_keys = [ordered[0][0], ordered[-1][0]]
    for key in fallback_keys:
        if key in seen:
            continue
        n_bits = _get_quantized_module_n_bits(name2mod[key])
        if (n_bits is not None) and (n_bits >= 8):
            seen.add(key)
            continue
        if verbose:
            print("[INFO] set 8bit for", key)
        _set_quantized_module_to_8bit(name2mod[key])
        seen.add(key)


def _build_trainable_mha_wrapper(
    attn_module: nn.MultiheadAttention,
    weight_quant_params: dict,
    act_quant_params: dict,
    out_proj_module_cls,
    out_proj_module_kwargs=None,
):
    return QuantMultiheadAttention(
        attn_module,
        weight_quant_params,
        act_quant_params,
        need_init=True,
        out_proj_module_cls=out_proj_module_cls,
        out_proj_module_kwargs=out_proj_module_kwargs
    )

class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, need_init=True):
        super().__init__()
        self.model = model
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params, need_init=need_init)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, need_init=True):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """

        for name, child_module in module.named_children():

            if isinstance(child_module, nn.MultiheadAttention):
                setattr(
                    module,
                    name,
                    QuantMultiheadAttention(
                        child_module,
                        weight_quant_params,
                        act_quant_params,
                        need_init=need_init
                    )
                )
                continue

            if isinstance(child_module, (nn.Conv1d, nn.Linear)) and 'skip' not in name and 'op' not in name:
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params, need_init=need_init))

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params, need_init=need_init)

    def add_act_quantizers(self, target_keys, act_params):
        name2mod = {n:m for n,m in self.model.named_modules() if isinstance(m, QuantModule)}
        for key in target_keys:
            if key in name2mod:
                m = name2mod[key]
                if m.act_quantizer is None:   # 없는 경우에만 생성
                    m.act_quantizer = UniformAffineQuantizer(**act_params, need_init=True)
                    m.use_act_quant = True


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, QuantMultiheadAttention)):  ## remove BaseQuantBlock
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def set_first_last_layer_to_8bit(self):
        _apply_first_last_8bit_overrides(
            self.model,
            quant_types=(QuantModule, QuantMultiheadAttention),
            verbose=False
        )

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True

class QuantModelLoRA(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, num_steps=100):
        super().__init__()
        # search_fold_and_remove_bn(model)
        self.model = model
        self.num_steps = num_steps
        self._t_norm = None       # 현재 배치의 정규화된 t를 보관
        self.count = 0
        self.total_count = 47 ## 265 for EfficientDM 
        self.special_module_count_list = [7,8,23,47] ## modify here for different dataset (imagenet or lsun) 
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv1d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv1d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None

        for name, child_module in module.named_children():
            if isinstance(child_module, nn.MultiheadAttention):
                self.count += 1
                if self.count in self.special_module_count_list:
                    out_proj_module_cls = QuantModule
                    out_proj_module_kwargs = None
                else:
                    out_proj_module_cls = QuantModuleLoRA
                    out_proj_module_kwargs = {'num_steps': self.num_steps}
                setattr(
                    module,
                    name,
                    _build_trainable_mha_wrapper(
                        child_module,
                        weight_quant_params,
                        act_quant_params,
                        out_proj_module_cls=out_proj_module_cls,
                        out_proj_module_kwargs=out_proj_module_kwargs
                    )
                )
                continue

            if isinstance(child_module, (nn.Conv1d, nn.Linear)) and 'skip' not in name and 'op' not in name:
                self.count += 1
                if self.count in self.special_module_count_list:
                    qm = QuantModule(child_module, weight_quant_params, act_quant_params, need_init=True)
                else:
                    qm = QuantModuleLoRA(child_module, weight_quant_params, act_quant_params, num_steps=self.num_steps)
                setattr(module, name, qm)
            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModuleLoRA, QuantModule, QuantMultiheadAttention)):  ## remove BaseQuantBlock
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, t, global_cond=None, **kwargs):
        out = self.model(x, t, global_cond=global_cond, **kwargs)
        return out
    
    def set_first_last_layer_to_8bit(self):
        _apply_first_last_8bit_overrides(
            self.model,
            quant_types=(QuantModuleLoRA, QuantModule, QuantMultiheadAttention),
            verbose=True
        )

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, (QuantModuleLoRA, QuantModule, QuantMultiheadAttention)):
                module_list += [m]
        module_list[-1].disable_act_quant = True

class QuantModelLoRATaDA(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, num_steps=100):
        super().__init__()
        # search_fold_and_remove_bn(model)
        self.model = model
        self.num_steps = num_steps
        self.total_bins = num_steps
        self._t_norm = None       # 현재 배치의 정규화된 t를 보관
        self._t_bin = None
        self.count = 0
        self.total_count = 47 ## 265 for EfficientDM 
        self.special_module_count_list = [7,8,23,47] ## modify here for different dataset (imagenet or lsun) 
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv1d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv1d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None

        for name, child_module in module.named_children():
            if isinstance(child_module, nn.MultiheadAttention):
                self.count += 1
                if self.count in self.special_module_count_list:
                    out_proj_module_cls = QuantModule
                    out_proj_module_kwargs = None
                else:
                    out_proj_module_cls = QuantModuleLoRATaDA
                    out_proj_module_kwargs = None

                qm = _build_trainable_mha_wrapper(
                    child_module,
                    weight_quant_params,
                    act_quant_params,
                    out_proj_module_cls=out_proj_module_cls,
                    out_proj_module_kwargs=out_proj_module_kwargs
                )
                if isinstance(qm.out_proj, QuantModuleLoRATaDA):
                    object.__setattr__(qm.out_proj, "t_provider_ref", weakref.ref(self))
                setattr(module, name, qm)
                continue

            if isinstance(child_module, (nn.Conv1d, nn.Linear)) and 'skip' not in name and 'op' not in name:
                self.count += 1
                if self.count in self.special_module_count_list:
                    qm = QuantModule(child_module, weight_quant_params, act_quant_params, need_init=True)
                else:
                    qm = QuantModuleLoRATaDA(child_module, weight_quant_params, act_quant_params)

                object.__setattr__(qm, "t_provider_ref", weakref.ref(self))
                setattr(module, name, qm)

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModuleLoRATaDA, QuantModule, QuantMultiheadAttention)):  ## remove BaseQuantBlock
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, t, global_cond=None, **kwargs): # qnn(calib data) 할 때 한번 
        # after (분기 제거)
        t = t.to(dtype=torch.long).reshape(-1)                          # 스칼라도 (1,)로 정규화
        den = (self.num_steps - 1) if (self.num_steps > 1) else 1
        self._t_bin = t.clamp(min=0, max=self.total_bins - 1)
        self._t_norm = (t.to(torch.float32) / float(den)).unsqueeze(1)  # [B,1]
        out = self.model(x, t, global_cond=global_cond)
        self._t_norm = None
        self._t_bin = None
        return out
    
    def set_first_last_layer_to_8bit(self):
        _apply_first_last_8bit_overrides(
            self.model,
            quant_types=(QuantModuleLoRATaDA, QuantModule, QuantMultiheadAttention),
            verbose=True
        )

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, (QuantModuleLoRATaDA, QuantModule, QuantMultiheadAttention)):
                module_list += [m]
        module_list[-1].disable_act_quant = True


class QuantModelLoRATaDA_org(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, num_steps=100):
        super().__init__()
        # search_fold_and_remove_bn(model)
        self.model = model
        self.num_steps = num_steps
        self._t_norm = None
        self.base_weight_bits = int(weight_quant_params.get('n_bits', 8))
        self.special_module_name_list = set(SPECIAL_8BIT_MODULE_KEYS)
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        print("[INFO] quantmodel is quantmodelloratada org!!")

    def _should_use_plain_quant_module(self, full_name: str):
        if (self.base_weight_bits == 4) and (full_name in self.special_module_name_list):
            return True
        return False

    def quant_module_refactor(
        self,
        module: nn.Module,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        prefix: str = "",
    ):
        """
        Recursively replace the normal conv1d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv1d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None

        for name, child_module in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child_module, nn.MultiheadAttention):
                qm = _build_trainable_mha_wrapper(
                    child_module,
                    weight_quant_params,
                    act_quant_params,
                    out_proj_module_cls=QuantModuleLoRATaDA_org,
                )
                if isinstance(getattr(qm, 'out_proj', None), QuantModuleLoRATaDA_org):
                    object.__setattr__(qm.out_proj, 't_provider_ref', weakref.ref(self))
                setattr(module, name, qm)
                continue

            if isinstance(child_module, (nn.Conv1d, nn.Linear)) and 'skip' not in name and 'op' not in name:
                if self._should_use_plain_quant_module(full_name):
                    qm = QuantModule(child_module, weight_quant_params, act_quant_params, need_init=True)
                else:
                    qm = QuantModuleLoRATaDA_org(child_module, weight_quant_params, act_quant_params)

                if isinstance(qm, QuantModuleLoRATaDA_org):
                    object.__setattr__(qm, 't_provider_ref', weakref.ref(self))
                setattr(module, name, qm)

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(
                    child_module,
                    weight_quant_params,
                    act_quant_params,
                    prefix=full_name,
                )

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModuleLoRATaDA_org, QuantModule, QuantMultiheadAttention)):  ## remove BaseQuantBlock
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, t, global_cond=None, **kwargs):
        t = t.to(dtype=torch.long).reshape(-1)
        den = (self.num_steps - 1) if (self.num_steps > 1) else 1
        self._t_norm = (t.to(torch.float32) / float(den)).unsqueeze(1)  # [B,1]
        out = self.model(x, t, global_cond=global_cond, **kwargs)
        self._t_norm = None
        return out

    def set_first_last_layer_to_8bit(self):
        _apply_first_last_8bit_overrides(
            self.model,
            quant_types=(QuantModuleLoRATaDA_org, QuantModule, QuantMultiheadAttention),
            verbose=True
        )

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, (QuantModuleLoRATaDA_org, QuantModule, QuantMultiheadAttention)):
                module_list += [m]
        module_list[-1].disable_act_quant = True