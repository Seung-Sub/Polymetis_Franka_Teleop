import torch.nn as nn 
import torch 
from quant.quant_layer import UniformAffineQuantizer
import torch.nn.functional as F
    
class QDQLinear(nn.Module): 
    def __init__(self,
                 linear: nn.Linear,
                 act_scale: float,
                 act_zp: int, 
                 w_scales: torch.Tensor,
                 w_zps: torch.Tensor,
                 axis: int = 0,
                 n_bits: int = 8): 
        
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.linear = linear         
        self.axis = axis  
        self.n_bits = n_bits
        out_features, in_features = linear.weight.shape
        self.factor = 2 ** (8 - self.n_bits)

        act_scale_refactor = act_scale / self.factor 
        self.register_buffer("act_zp", torch.tensor(act_zp, dtype=torch.int32))
        self.register_buffer("act_scale", torch.tensor(act_scale_refactor, dtype=torch.float32))

        assert w_scales.numel() == out_features
        assert w_zps.numel()    == out_features

        w_scales_refactor = w_scales.detach().float().view(-1) / self.factor
        self.register_buffer("w_zps",    w_zps.to(torch.int32))
        self.register_buffer("w_scales", w_scales_refactor.to(torch.float32))

        self.register_buffer("weight_fp", linear.weight.detach().float())
        if linear.bias is not None:
            self.register_buffer("bias_fp", linear.bias.detach().float())
        else:
            self.bias_fp = None

    def forward(self, x):
        x_dq = torch.fake_quantize_per_tensor_affine(
            x,
            self.act_scale,
            self.act_zp,
            quant_min=-128,
            quant_max=127
        )
        w_dq = torch.fake_quantize_per_channel_affine(
            self.weight_fp,
            self.w_scales,
            self.w_zps,
            self.axis,
            quant_min=-128,
            quant_max=127
        )
        
        x_dq = x_dq.to(w_dq.device)

        y = F.linear(x_dq,
                     w_dq,
                     self.bias_fp
        )
        return y

class QDQConv1d(nn.Module):
    def __init__(self,
                 conv: nn.Conv1d,
                 act_scale: float,
                 act_zp: int,
                 w_scales: torch.Tensor,
                 w_zps: torch.Tensor,
                 axis: int = 0,
                 n_bits: int = 8):
        
        super().__init__()
        assert isinstance(conv, nn.Conv1d)
        self.conv = conv
        self.axis = axis
        self.n_bits = n_bits
        self.factor = 2 ** (8 - self.n_bits) 

        act_scale_refactor = act_scale / self.factor
        self.register_buffer("act_zp", torch.tensor(act_zp, dtype=torch.int32))
        self.register_buffer("act_scale", torch.tensor(act_scale_refactor, dtype=torch.float32))

        w_scales_refactor = w_scales.detach().float() / self.factor 
        self.register_buffer("w_zps", w_zps.to(torch.int32))
        self.register_buffer("w_scales", w_scales_refactor.to(torch.float32))

        self.register_buffer("weight_fp", conv.weight.detach().float())
        if conv.bias is not None:
            self.register_buffer("bias_fp", conv.bias.detach().float())
        else:
            self.bias_fp = None 

    def forward(self, x): 
        x_dq = torch.fake_quantize_per_tensor_affine(
            x,
            self.act_scale,
            self.act_zp,           
            quant_min=-128,
            quant_max=127
        )
        w_dq = torch.fake_quantize_per_channel_affine(
            self.weight_fp,
            self.w_scales,
            self.w_zps,
            self.axis,
            quant_min=-128,
            quant_max=127
        )

        x_dq = x_dq.to(w_dq.device)

        y = F.conv1d(x_dq,
                     w_dq,
                     self.bias_fp,
                     stride = self.conv.stride,
                     padding = self.conv.padding,
                     dilation = self.conv.dilation, 
                     groups = self.conv.groups,
        )
        return y 

class QDQLinear_TaDA(nn.Module): 
    def __init__(self,
                 linear: nn.Linear,
                 act_scale: float,
                 act_zp: int,
                 w_scales: torch.Tensor,
                 w_zps: torch.Tensor,
                 axis: int = 0, 
                 n_bits: int = 8):  
        
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.linear = linear
        self.axis = axis 
        self.n_bits = n_bits
        out_features, in_features = linear.weight.shape
        self.factor = 2 ** (8 - self.n_bits) 

        act_scale_refactor = act_scale / self.factor 
        self.register_buffer("act_zp", torch.tensor(act_zp, dtype=torch.int32))
        self.register_buffer("act_scale", torch.tensor(act_scale_refactor, dtype=torch.float32))

        assert w_scales.numel() == out_features
        assert w_zps.numel()    == out_features

        w_scales_refactor = w_scales.detach().float().view(-1) / self.factor
        self.register_buffer("w_zps", w_zps.to(torch.int32))
        self.register_buffer("w_scales", w_scales_refactor.to(torch.float32))

        self.register_buffer("weight_fp", linear.weight.detach())
        if linear.bias is not None:
            self.register_buffer("bias_fp", linear.bias.detach())
        else:
            self.bias_fp = None

    def forward(self, x):
        w_dq = torch.fake_quantize_per_channel_affine(
            self.weight_fp,
            self.w_scales,
            self.w_zps,
            self.axis,
            quant_min=-128,
            quant_max=127,
        )
        y = F.linear(x,
                     w_dq,
                     self.bias_fp
        )
        return y

class QDQConv1d_TaDA(nn.Module):
    def __init__(self,
                 conv: nn.Conv1d,
                 act_scale: float,
                 act_zp: int,
                 w_scales: torch.Tensor,
                 w_zps: torch.Tensor,
                 axis: int = 0,
                 n_bits: int = 8):
        
        super().__init__()
        assert isinstance(conv, nn.Conv1d)
        self.conv = conv
        self.axis = axis
        self.n_bits = n_bits
        self.factor = 2 ** (8 - self.n_bits)

        act_scale_refactor = act_scale / self.factor
        self.register_buffer("act_zp", torch.tensor(act_zp, dtype=torch.int32))
        self.register_buffer("act_scale", torch.tensor(act_scale_refactor, dtype=torch.float32))

        w_scales_refactor = w_scales.detach().float() / self.factor 
        self.register_buffer("w_zps", w_zps.to(torch.int32))
        self.register_buffer("w_scales", w_scales_refactor.to(torch.float32))

        self.register_buffer("weight_fp", conv.weight.detach())
        if conv.bias is not None:
            self.register_buffer("bias_fp", conv.bias.detach())
        else:
            self.bias_fp = None

    def forward(self, x):
        w_dq = torch.fake_quantize_per_channel_affine(
            self.weight_fp,
            self.w_scales,
            self.w_zps,
            self.axis,
            quant_min=-128,
            quant_max=127,
        )
        y = F.conv1d(x,
                     w_dq,   
                     self.bias_fp,
                     stride=self.conv.stride,
                     padding=self.conv.padding,
                     dilation=self.conv.dilation,
                     groups=self.conv.groups,
        )
        return y