import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import math
        
class TScaler(nn.Module):
    def __init__(self, hidden=16, s_min=0.25, s_max=4.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )
        self.s_min = s_min
        self.s_max = s_max
    def forward(self, t_norm):                     
        dev = next(self.net.parameters()).device
        if t_norm.device != dev: 
            t_norm = t_norm.to(dev)
        s = F.softplus(self.net(t_norm)) + 1e-4    
        return torch.clamp(s, self.s_min, self.s_max)
    
class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    # y = x.detach().round()
    # return x + (y-x).detach()
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, weight_tensor = None, need_init=True):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

        if weight_tensor is not None:
            self.inited = True
            if len(weight_tensor.shape) == 4:
                self.delta = nn.Parameter(torch.randn(size=(weight_tensor.shape[0], 1, 1, 1))) ## removed requires_grad=False here
                self.zero_point = nn.Parameter(torch.randn(size=(weight_tensor.shape[0], 1, 1, 1)))
            elif len(weight_tensor.shape) == 3: # for diffusion policy, Conv1d
                self.delta = nn.Parameter(torch.randn(size=(weight_tensor.shape[0],1, 1)))
                self.zero_point = nn.Parameter(torch.randn(weight_tensor.shape[0], 1, 1))
            elif len(weight_tensor.shape) == 2:
                self.delta = nn.Parameter(torch.randn(size=(weight_tensor.shape[0], 1)))
                self.zero_point = nn.Parameter(torch.randn(size=(weight_tensor.shape[0], 1)))           
            else: 
                print(weight_tensor.shape)
                raise ValueError('shape not implemented')
        else:
            self.inited = not need_init # use this when quantizing models
            self.delta = nn.Parameter(torch.tensor(0.005)) ## removed requires_grad=False here
            self.zero_point = nn.Parameter(torch.tensor(0.0))

    def _signed_range(self):
        # e.g., 8-bit -> -128 ~ 127, 4-bit -> -8 ~ 7
        qmin = -(1<<(self.n_bits-1)) 
        qmax = (1<<(self.n_bits-1))-1 
        return qmin, qmax 
    
    def clipping(self, x, lower, upper):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)
        return x
    
    def forward(self, x: torch.Tensor):
        if self.inited is False:
            delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
            if not isinstance(zero_point, torch.Tensor):
                zero_point = torch.tensor(float(zero_point))
            self.delta = torch.nn.Parameter(delta)
            self.zero_point = torch.nn.Parameter(zero_point)
            self.inited = True
        # start quantization
        if self.sym: # signed-symmetric: zp=0, clamp to [-q_max, q_max]
            qmin, qmax = self._signed_range()
            scale = self.delta.abs().clamp_min(1e-8)
            scale = scale.to(device=x.device, dtype=x.dtype)
            x_int   = round_ste(x / scale)        # zp = 0
            x_quant = self.clipping(x_int, qmin, qmax)  # [-128, 127]
            x_dequant = x_quant * scale
        else: # asymmetric 
            x_int = round_ste(x / self.delta) + self.zero_point # x = weight_tensor ([C_out, C_in, K (kernel)])
            x_quant = self.clipping(x_int, 0, self.n_levels - 1) # modified here to replace torch.clamp for gradient prop
            # x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        eps = 1e-8
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 3: 
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]

            if self.sym: 
                # symmetric path 
                qmin, qmax = self._signed_range()
                delta = (x_max/qmax).clamp(min=eps)
                zero_point = torch.zeros_like(delta)
            else:
                # asymmetric path 
                qmin, qmax_u = 0, self.n_levels-1
                x_min = -x_max 
                x_max_u = x_max 
                delta = ((x_max_u-x_min)/(qmax_u-qmin)).clamp(min=eps)
                zero_point = (-x_min/delta).round().clamp(qmin, qmax_u)
                # delta = x_max.clone()
                # zero_point = x_max.clone()

            ## comment below for faster initialization in inference
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)

            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 3 : 
                delta = delta.view(-1, 1, 1)
                zero_point = zero_point.view(-1,1,1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                if self.sym:
                    a = x.detach().abs().max()
                    # x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
                    qmin, qmax = self._signed_range()
                    delta = torch.tensor(max(float(a)/qmax, eps)).type_as(x)
                    zero_point = torch.tensor(0.0).type_as(x)
                    return delta, zero_point
                else: 
                    delta = float(x_max - x_min) / (self.n_levels - 1)
                    if delta < 1e-8:
                        warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                        delta = 1e-8

                    zero_point = round(-x_min / delta)
                    delta = torch.tensor(delta).type_as(x)
                    return delta, zero_point 

            elif self.scale_method == 'mse':
                if self.sym: 
                    qmin, qmax = self._signed_range()  
                    a0 = x.detach().abs().max() 
                    best_score = float('inf')
                    best_a = a0 

                    for i in range(80): # a를 줄여가면서 mse 최소 탐색 
                        a = a0 * (1.0-0.01*i)
                        delta_i = max(float(a)/qmax ,eps)   

                        x_int = torch.round(x/delta_i)
                        x_q = x_int.clamp(qmin, qmax)
                        x_hat = x_q * delta_i
                        score = F.mse_loss(x_hat, x)
                        if score < best_score:
                            best_score = score 
                            best_a = a
                    delta = torch.tensor(max(float(best_a)/qmax, eps)).type_as(x)
                    zero_point = torch.tensor(0.0).type_as(x)
                    return delta, zero_point
                else: 
                    x_max = x.max()
                    x_min = x.min()
                    best_score = 1e+10
                    for i in range(80):
                        new_max = x_max * (1.0 - (i * 0.01))
                        new_min = x_min * (1.0 - (i * 0.01))
                        x_q = self.quantize(x, new_max, new_min)
                        # L_p norm minimization as described in LAPQ
                        # https://arxiv.org/abs/1911.07190
                        score = lp_loss(x, x_q, p=2.4, reduction='all')
                        if score < best_score:
                            best_score = score
                            delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                            zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


def _build_attention_out_proj(
    org_module: nn.Linear,
    weight_quant_params: dict,
    act_quant_params: dict,
    disable_act_quant: bool,
    need_init: bool,
    out_proj_module_cls=None,
    out_proj_module_kwargs=None,
):
    if out_proj_module_cls is None:
        out_proj_module_cls = QuantModule
    if out_proj_module_kwargs is None:
        out_proj_module_kwargs = {}

    if out_proj_module_cls is QuantModule:
        return out_proj_module_cls(
            org_module,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
            disable_act_quant=disable_act_quant,
            need_init=need_init
        )

    return out_proj_module_cls(
        org_module,
        weight_quant_params=weight_quant_params,
        act_quant_params=act_quant_params,
        **out_proj_module_kwargs
    )


class QuantMultiheadAttention(nn.Module):
    """
    Quantized wrapper for nn.MultiheadAttention.
    The packed q/k/v projection weights live as parameters on the attention module
    itself, so they never hit QuantModule unless we wrap the whole block.
    """
    def __init__(self, org_module: nn.MultiheadAttention, weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, need_init=True,
                 out_proj_module_cls=None, out_proj_module_kwargs=None):
        super().__init__()

        self.embed_dim = org_module.embed_dim
        self.kdim = org_module.kdim
        self.vdim = org_module.vdim
        self.num_heads = org_module.num_heads
        self.dropout = org_module.dropout
        self.head_dim = org_module.head_dim
        self.batch_first = getattr(org_module, 'batch_first', False)
        self.add_zero_attn = org_module.add_zero_attn
        self._qkv_same_embed_dim = org_module._qkv_same_embed_dim

        if org_module.in_proj_weight is not None:
            self.in_proj_weight = org_module.in_proj_weight
            self.org_in_proj_weight = org_module.in_proj_weight.data.clone()
        else:
            self.register_parameter('in_proj_weight', None)
            self.org_in_proj_weight = None

        if org_module.in_proj_bias is not None:
            self.in_proj_bias = org_module.in_proj_bias
            self.org_in_proj_bias = org_module.in_proj_bias.data.clone()
        else:
            self.register_parameter('in_proj_bias', None)
            self.org_in_proj_bias = None

        if getattr(org_module, 'q_proj_weight', None) is not None:
            self.q_proj_weight = org_module.q_proj_weight
            self.org_q_proj_weight = org_module.q_proj_weight.data.clone()
        else:
            self.register_parameter('q_proj_weight', None)
            self.org_q_proj_weight = None

        if getattr(org_module, 'k_proj_weight', None) is not None:
            self.k_proj_weight = org_module.k_proj_weight
            self.org_k_proj_weight = org_module.k_proj_weight.data.clone()
        else:
            self.register_parameter('k_proj_weight', None)
            self.org_k_proj_weight = None

        if getattr(org_module, 'v_proj_weight', None) is not None:
            self.v_proj_weight = org_module.v_proj_weight
            self.org_v_proj_weight = org_module.v_proj_weight.data.clone()
        else:
            self.register_parameter('v_proj_weight', None)
            self.org_v_proj_weight = None

        if org_module.bias_k is not None:
            self.bias_k = org_module.bias_k
        else:
            self.register_parameter('bias_k', None)

        if org_module.bias_v is not None:
            self.bias_v = org_module.bias_v
        else:
            self.register_parameter('bias_v', None)

        self.out_proj = _build_attention_out_proj(
            org_module.out_proj,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
            disable_act_quant=disable_act_quant,
            need_init=need_init,
            out_proj_module_cls=out_proj_module_cls,
            out_proj_module_kwargs=out_proj_module_kwargs
        )

        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        self.ignore_reconstruction = False

        if self._qkv_same_embed_dim:
            if not need_init:
                self.in_proj_weight_quantizer = UniformAffineQuantizer(
                    **weight_quant_params, weight_tensor=self.in_proj_weight
                )
            else:
                self.in_proj_weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        else:
            if not need_init:
                self.q_proj_weight_quantizer = UniformAffineQuantizer(
                    **weight_quant_params, weight_tensor=self.q_proj_weight
                )
                self.k_proj_weight_quantizer = UniformAffineQuantizer(
                    **weight_quant_params, weight_tensor=self.k_proj_weight
                )
                self.v_proj_weight_quantizer = UniformAffineQuantizer(
                    **weight_quant_params, weight_tensor=self.v_proj_weight
                )
            else:
                self.q_proj_weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
                self.k_proj_weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
                self.v_proj_weight_quantizer = UniformAffineQuantizer(**weight_quant_params)

        self.query_act_quantizer = UniformAffineQuantizer(**act_quant_params, need_init=need_init)
        self.key_act_quantizer = UniformAffineQuantizer(**act_quant_params, need_init=need_init)
        self.value_act_quantizer = UniformAffineQuantizer(**act_quant_params, need_init=need_init)
        self.extra_repr = org_module.extra_repr

    @staticmethod
    def _move_to_ref(tensor: torch.Tensor, ref: torch.Tensor):
        if tensor is None:
            return None
        if tensor.device != ref.device:
            tensor = tensor.to(ref.device)
        if tensor.dtype != ref.dtype:
            tensor = tensor.to(ref.dtype)
        return tensor

    def _project_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        if self._qkv_same_embed_dim:
            if self.use_weight_quant:
                in_proj_weight = self.in_proj_weight_quantizer(self.in_proj_weight)
                in_proj_bias = self.in_proj_bias
            else:
                in_proj_weight = self.org_in_proj_weight
                in_proj_bias = self.org_in_proj_bias

            in_proj_weight = self._move_to_ref(in_proj_weight, query)
            in_proj_bias = self._move_to_ref(in_proj_bias, query)

            if query.dtype != in_proj_weight.dtype:
                query = query.to(in_proj_weight.dtype)
                key = key.to(in_proj_weight.dtype)
                value = value.to(in_proj_weight.dtype)

            w_q, w_k, w_v = in_proj_weight.chunk(3, dim=0)
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3, dim=0)
        else:
            if self.use_weight_quant:
                w_q = self.q_proj_weight_quantizer(self.q_proj_weight)
                w_k = self.k_proj_weight_quantizer(self.k_proj_weight)
                w_v = self.v_proj_weight_quantizer(self.v_proj_weight)
                in_proj_bias = self.in_proj_bias
            else:
                w_q = self.org_q_proj_weight
                w_k = self.org_k_proj_weight
                w_v = self.org_v_proj_weight
                in_proj_bias = self.org_in_proj_bias

            w_q = self._move_to_ref(w_q, query)
            w_k = self._move_to_ref(w_k, key)
            w_v = self._move_to_ref(w_v, value)
            in_proj_bias = self._move_to_ref(in_proj_bias, query)

            if query.dtype != w_q.dtype:
                query = query.to(w_q.dtype)
            if key.dtype != w_k.dtype:
                key = key.to(w_k.dtype)
            if value.dtype != w_v.dtype:
                value = value.to(w_v.dtype)

            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3, dim=0)

        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)
        return q, k, v

    def _reshape_heads(self, x: torch.Tensor):
        bsz, seq_len, _ = x.shape
        return x.contiguous().view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _apply_attn_mask(self, scores: torch.Tensor, attn_mask: torch.Tensor, bsz: int, tgt_len: int, src_len: int):
        if attn_mask is None:
            return scores

        attn_mask = attn_mask.to(device=scores.device)
        if attn_mask.dtype in (torch.bool, torch.uint8):
            attn_mask = attn_mask.bool()
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.view(1, 1, tgt_len, src_len)
            elif attn_mask.dim() == 3:
                if attn_mask.shape[0] == bsz * self.num_heads:
                    attn_mask = attn_mask.view(bsz, self.num_heads, tgt_len, src_len)
                elif attn_mask.shape[0] == bsz:
                    attn_mask = attn_mask.unsqueeze(1)
                else:
                    raise ValueError(f"Unsupported attention mask shape: {tuple(attn_mask.shape)}")
            else:
                raise ValueError(f"Unsupported attention mask rank: {attn_mask.dim()}")
            return scores.masked_fill(attn_mask, float('-inf'))

        attn_mask = attn_mask.to(dtype=scores.dtype)
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.view(1, 1, tgt_len, src_len)
        elif attn_mask.dim() == 3:
            if attn_mask.shape[0] == bsz * self.num_heads:
                attn_mask = attn_mask.view(bsz, self.num_heads, tgt_len, src_len)
            elif attn_mask.shape[0] == bsz:
                attn_mask = attn_mask.unsqueeze(1)
            else:
                raise ValueError(f"Unsupported attention mask shape: {tuple(attn_mask.shape)}")
        else:
            raise ValueError(f"Unsupported attention mask rank: {attn_mask.dim()}")
        return scores + attn_mask

    def _apply_key_padding_mask(self, scores: torch.Tensor, key_padding_mask: torch.Tensor):
        if key_padding_mask is None:
            return scores

        if key_padding_mask.dim() == 1:
            key_padding_mask = key_padding_mask.unsqueeze(0)
        key_padding_mask = key_padding_mask.to(device=scores.device)

        if key_padding_mask.dtype in (torch.bool, torch.uint8):
            return scores.masked_fill(key_padding_mask.bool()[:, None, None, :], float('-inf'))

        return scores + key_padding_mask.to(dtype=scores.dtype)[:, None, None, :]

    def bitwidth_refactor(self, refactored_bit: int):
        if self._qkv_same_embed_dim:
            self.in_proj_weight_quantizer.bitwidth_refactor(refactored_bit)
        else:
            self.q_proj_weight_quantizer.bitwidth_refactor(refactored_bit)
            self.k_proj_weight_quantizer.bitwidth_refactor(refactored_bit)
            self.v_proj_weight_quantizer.bitwidth_refactor(refactored_bit)

        self.query_act_quantizer.bitwidth_refactor(refactored_bit)
        self.key_act_quantizer.bitwidth_refactor(refactored_bit)
        self.value_act_quantizer.bitwidth_refactor(refactored_bit)
        self.out_proj.weight_quantizer.bitwidth_refactor(refactored_bit)
        if getattr(self.out_proj, 'act_quantizer', None) is not None:
            self.out_proj.act_quantizer.bitwidth_refactor(refactored_bit)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        if hasattr(self.out_proj, 'set_quant_state'):
            self.out_proj.set_quant_state(weight_quant, act_quant)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: torch.Tensor = None, need_weights: bool = True,
                attn_mask: torch.Tensor = None, average_attn_weights: bool = True,
                is_causal: bool = False):
        if self.bias_k is not None or self.bias_v is not None or self.add_zero_attn:
            raise NotImplementedError(
                "QuantMultiheadAttention currently supports the standard decoder attention path "
                "without bias_k/bias_v/add_zero_attn."
            )

        is_batched = query.dim() == 3
        if not is_batched:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

        if not self.batch_first and is_batched:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        if self.use_act_quant and not self.disable_act_quant:
            query = self.query_act_quantizer(query)
            key = self.key_act_quantizer(key)
            value = self.value_act_quantizer(value)

        q, k, v = self._project_qkv(query, key, value)

        bsz, tgt_len, _ = q.shape
        src_len = k.shape[1]

        q = self._reshape_heads(q) * (1.0 / math.sqrt(self.head_dim))
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(tgt_len, src_len, device=attn_scores.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask.view(1, 1, tgt_len, src_len), float('-inf'))

        attn_scores = self._apply_attn_mask(attn_scores, attn_mask, bsz, tgt_len, src_len)
        attn_scores = self._apply_key_padding_mask(attn_scores, key_padding_mask)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if not self.batch_first and is_batched:
            attn_output = attn_output.transpose(0, 1)

        if not is_batched:
            attn_output = attn_output.squeeze(0)

        if not need_weights:
            return attn_output, None

        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)

        if not is_batched:
            attn_weights = attn_weights.squeeze(0)

        return attn_output, attn_weights
    
class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv1d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, se_module=None, need_init=True, enable_act_quant=False):
        super(QuantModule, self).__init__()

        if isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight #.half()
        self.org_weight = org_module.weight.data.clone() #.half()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias =org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        if not need_init:
            self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params, weight_tensor=self.weight)
        else:
            self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params) ## delta need to be inited

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params, need_init=need_init)
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight) # .half()
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        if self.use_act_quant:
            input = self.act_quantizer(input)

        dev = input.device
        if isinstance(weight, torch.Tensor) and weight.device != dev:
            weight = weight.to(dev)
        if bias is not None and isinstance(bias, torch.Tensor) and bias.device != dev:
            bias = bias.to(dev)
        if input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        if bias is not None and bias.dtype != weight.dtype:
            bias = bias.to(weight.dtype)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs) # origin
        if self.se_module is not None:
            out = self.se_module(out)
        out = self.activation_function(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def bitwidth_refactor(self, refactored_bit: int):
        self.weight_quantizer.bitwidth_refactor(refactored_bit)
        self.act_quantizer.bitwidth_refactor(refactored_bit)


class QuantModuleLoRA(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv1d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, num_steps=100):
        super(QuantModuleLoRA, self).__init__()
        if isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        
        self.ori_shape = org_module.weight.shape
        self.size_scale = int(8 // weight_quant_params['n_bits'])
        
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()

        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = False
        # initialize quantizer
        self.intn_dequantizer = None ## to be inited
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params, need_init=True)
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.extra_repr = org_module.extra_repr

        ## add lora here
        r = 64
        lora_dropout = 0.0
        if lora_dropout > 0.0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = nn.Identity()
        if isinstance(org_module, nn.Linear) and self.weight_quantizer.n_bits <= 8:
            self.loraA = nn.Linear(org_module.in_features, r, bias=False)
            self.loraB = nn.Linear(r, org_module.out_features, bias=False)
            nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5)) 
            nn.init.zeros_(self.loraB.weight)
        elif isinstance(org_module, nn.Conv1d) and self.weight_quantizer.n_bits <= 8:
            self.loraA = nn.Conv1d(org_module.in_channels, r, org_module.kernel_size, org_module.stride, org_module.padding, \
                                   org_module.dilation, org_module.groups, bias=False)
            self.loraB = nn.Conv1d(r, org_module.out_channels, 1, bias=False)
            nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5)) 
            nn.init.zeros_(self.loraB.weight)

    def forward(self, input: torch.Tensor):
        orig_weight = self.weight 
        if self.fwd_func is F.linear:
            E = torch.eye(orig_weight.shape[1], device=input.device)
            lora_weight = self.loraB(self.loraA(self.lora_dropout_layer(E)))
            lora_weight = lora_weight.T
            weight = orig_weight + lora_weight
        elif self.fwd_func is F.conv1d: # lora kernel's shape : (out_channels, in_channels, k)
            A = self.loraA.weight                     # (r, in, k)
            B = self.loraB.weight.squeeze(-1)         # (out, r)
            r, in_c, k = A.shape
            out_c = B.shape[0]
            lora_weight = (B @ A.reshape(r, -1)).reshape(out_c, in_c, k)  # (out, in, k)
            weight = orig_weight + lora_weight
        else:
            weight = orig_weight

        if self.use_weight_quant:
            weight = self.weight_quantizer(weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        if self.use_act_quant:
            input = self.act_quantizer(input)
        dev = input.device
        if isinstance(weight, torch.Tensor) and weight.device != dev: 
            weight = weight.to(dev)
        if bias is not None and isinstance(bias, torch.Tensor) and bias.device != dev: 
            bias = bias.to(dev)
        if input.dtype != weight.dtype: 
            input = input.to(weight.dtype)
        if bias is not None and bias.dtype != weight.dtype: 
            bias = bias.to(weight.dtype)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        
class QuantModuleLoRATaDA(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv1d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}):
        super(QuantModuleLoRATaDA, self).__init__()
        if isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        
        self.ori_shape = org_module.weight.shape
        self.size_scale = int(8 // weight_quant_params['n_bits'])
        
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()

        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.intn_dequantizer = None ## to be inited
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params, need_init=True)
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.extra_repr = org_module.extra_repr

        # === timestep-aware denoising adaptation ===
        self.t_scaler = TScaler(hidden=16, s_min=0.25, s_max=4.0)
        self.register_buffer("delta0", torch.tensor(0.005))   
        ## add lora here
        r = 64
        lora_dropout = 0.0
        if lora_dropout > 0.0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = nn.Identity()
        if isinstance(org_module, nn.Linear) and self.weight_quantizer.n_bits <= 8:
            self.loraA = nn.Linear(org_module.in_features, r, bias=False)
            self.loraB = nn.Linear(r, org_module.out_features, bias=False)
            nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5)) ## what's the use of a=math.sqrt(5)?
            nn.init.zeros_(self.loraB.weight)
        elif isinstance(org_module, nn.Conv1d) and self.weight_quantizer.n_bits <= 8:
            self.loraA = nn.Conv1d(org_module.in_channels, r, org_module.kernel_size, org_module.stride, org_module.padding, \
                                   org_module.dilation, org_module.groups, bias=False)
            self.loraB = nn.Conv1d(r, org_module.out_channels, 1, bias=False)
            nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5)) ## what's the use of a=math.sqrt(5)?
            nn.init.zeros_(self.loraB.weight)

    def forward(self, input: torch.Tensor):
        orig_weight = self.weight # self.intn_dequantizer(self.weight)
        if self.fwd_func is F.linear:
            E = torch.eye(orig_weight.shape[1], device=input.device)
            lora_weight = self.loraB(self.loraA(self.lora_dropout_layer(E)))
            lora_weight = lora_weight.T
            weight = orig_weight + lora_weight
            is_conv1d = False
        elif self.fwd_func is F.conv1d: # lora kernel's shape : (out_channels, in_channels, k)
            A = self.loraA.weight                     # (r, in, k)
            B = self.loraB.weight.squeeze(-1)         # (out, r)
            r, in_c, k = A.shape
            out_c = B.shape[0]
            lora_weight = (B @ A.reshape(r, -1)).reshape(out_c, in_c, k)  # (out, in, k)
            weight = orig_weight + lora_weight
            is_conv1d = True
        else:
            weight = orig_weight
            is_conv1d = False

        if self.use_weight_quant:
            # here
            weight = self.weight_quantizer(weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        if self.use_act_quant and not self.disable_act_quant:
            aq = self.act_quantizer

            prov = getattr(self, "t_provider_ref", None)
            parent = prov() if prov is not None else None 
            t_norm = getattr(parent, "_t_norm", None) if parent is not None else None
            t_bin = getattr(parent, "_t_bin", None) if parent is not None else None


            if t_norm is not None:
                B_in = input.shape[0]
                B0 = t_norm.size(0)
                if B_in != B0:
                    if B0 == 1:
                        t_norm = t_norm.expand(B_in, 1)
                        if t_bin is not None:
                            t_bin = t_bin.expand(B_in)
                    elif (B_in % B0) == 0:
                        rep = B_in // B0
                        t_norm = t_norm.repeat_interleave(rep, dim=0)
                        if t_bin is not None:
                            t_bin = t_bin.repeat_interleave(rep, dim=0)
                    else:
                        rep = (B_in + B0 - 1) // B0
                        t_norm = t_norm.repeat_interleave(rep, dim=0)[:B_in, :]
                        if t_bin is not None:
                            t_bin = t_bin.repeat_interleave(rep, dim=0)[:B_in]

                s = self.t_scaler(t_norm)  # [B,1]
                s_view = s.view(B_in, 1, 1) if input.dim() == 3 else s.view(B_in, 1)

                rec = getattr(self, "act_recorder", None)
                # print("[FINO] in quantlayer, t_bin: ", t_bin)
                if rec is not None and t_bin is not None:
                    rec.update("raw", t_bin, input)

                x = input * s_view
                if rec is not None and t_bin is not None:
                    rec.update("scaled", t_bin, x)

                with torch.no_grad():
                    aq.delta.data = self.delta0.to(aq.delta.device, dtype=aq.delta.dtype)
                    if hasattr(aq, "zero_point"):
                        aq.zero_point.data.zero_()

                xq = aq(x)
                input = xq / s_view

                if rec is not None and t_bin is not None:
                    rec.update("post", t_bin, input)
            else:
                input = aq(input)

        dev = input.device
        if isinstance(weight, torch.Tensor) and weight.device != dev: 
            weight = weight.to(dev)
        if bias is not None and isinstance(bias, torch.Tensor) and bias.device != dev: 
            bias = bias.to(dev)
        if input.dtype != weight.dtype: 
            input = input.to(weight.dtype)
        if bias is not None and bias.dtype != weight.dtype: 
            bias = bias.to(weight.dtype)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        
class QuantModuleLoRATaDA_org(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv1d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}):
        super(QuantModuleLoRATaDA_org, self).__init__()
        if isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        
        self.ori_shape = org_module.weight.shape
        self.size_scale = int(8 // weight_quant_params['n_bits'])
        
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()

        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = False
        # initialize quantizer
        self.intn_dequantizer = None ## to be inited
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params, need_init=True)
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.extra_repr = org_module.extra_repr

        # === timestep-aware denoising adaptation ===
        self.t_scaler = TScaler(hidden=16, s_min=0.25, s_max=4.0)
        self.register_buffer("delta0", torch.tensor(0.005))   
        print("[INFO] quantmodule is org!!!")    
        ## add lora here
        r = 64
        lora_dropout = 0.0
        if lora_dropout > 0.0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = nn.Identity()
        if isinstance(org_module, nn.Linear) and self.weight_quantizer.n_bits <= 8:
            self.loraA = nn.Linear(org_module.in_features, r, bias=False)
            self.loraB = nn.Linear(r, org_module.out_features, bias=False)
            nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5)) ## what's the use of a=math.sqrt(5)?
            nn.init.zeros_(self.loraB.weight)
        elif isinstance(org_module, nn.Conv1d) and self.weight_quantizer.n_bits <= 8:
            self.loraA = nn.Conv1d(org_module.in_channels, r, org_module.kernel_size, org_module.stride, org_module.padding, \
                                   org_module.dilation, org_module.groups, bias=False)
            self.loraB = nn.Conv1d(r, org_module.out_channels, 1, bias=False)
            nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5)) ## what's the use of a=math.sqrt(5)?
            nn.init.zeros_(self.loraB.weight)

    def forward(self, input: torch.Tensor):
        orig_weight = self.weight # self.intn_dequantizer(self.weight)
        if self.fwd_func is F.linear:
            E = torch.eye(orig_weight.shape[1], device=input.device)
            lora_weight = self.loraB(self.loraA(self.lora_dropout_layer(E)))
            lora_weight = lora_weight.T
            weight = orig_weight + lora_weight
            is_conv1d = False
        elif self.fwd_func is F.conv1d: # lora kernel's shape : (out_channels, in_channels, k)
            A = self.loraA.weight                     # (r, in, k)
            B = self.loraB.weight.squeeze(-1)         # (out, r)
            r, in_c, k = A.shape
            out_c = B.shape[0]
            lora_weight = (B @ A.reshape(r, -1)).reshape(out_c, in_c, k)  # (out, in, k)
            weight = orig_weight + lora_weight
            is_conv1d = True
        else:
            weight = orig_weight
            is_conv1d = False

        if self.use_weight_quant:
            # here
            weight = self.weight_quantizer(weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        if self.use_act_quant and not self.disable_act_quant:
            aq = self.act_quantizer

            t_norm = None
            ref = getattr(self, "t_provider_ref", None)
            if ref is not None:
                prov = ref()
                if prov is not None:
                    t_norm = getattr(prov, "_t_norm", None)
            if t_norm is not None:
                B_in = input.shape[0] # input batch size 
                B0 = t_norm.size(0)
                if B_in != B0: 
                    if B0 == 1: 
                        t_norm = t_norm.expand(B_in, 1)      # [B, hidden]
                    elif (B_in % B0) == 0:
                        rep = B_in // B0 
                        t_norm = t_norm.repeat_interleave(rep, dim=0)  # [B, hidden]
                    else: 
                        rep = (B_in + B0 -1) // B0
                        t_norm = t_norm.repeat_interleave(rep, dim=0)[:B_in, :]
                s = self.t_scaler(t_norm)                               # [B,1]
                if input.dim() == 3:        # Conv1d: [B,C,L]
                    s_view = s.view(B_in, 1, 1)
                else:                        # Linear: [..., In] 
                    s_view = s.view(B_in, 1)
                # pre-scale
                x = input * s_view
                with torch.no_grad():
                    aq.delta.data = self.delta0.to(aq.delta.device, dtype=aq.delta.dtype)
                    if hasattr(aq, "zero_point"):
                        aq.zero_point.data.zero_()
                xq = aq(x)                 
                input = xq / s_view         
            else: 
                input = aq(input)
        dev = input.device
        if isinstance(weight, torch.Tensor) and weight.device != dev: 
            weight = weight.to(dev)
        if bias is not None and isinstance(bias, torch.Tensor) and bias.device != dev: 
            bias = bias.to(dev)
        if input.dtype != weight.dtype: 
            input = input.to(weight.dtype)
        if bias is not None and bias.dtype != weight.dtype: 
            bias = bias.to(weight.dtype)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def bitwidth_refactor(self, refactored_bit: int):
        self.weight_quantizer.bitwidth_refactor(refactored_bit)
        if getattr(self, 'act_quantizer', None) is not None:
            self.act_quantizer.bitwidth_refactor(refactored_bit)
        