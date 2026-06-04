import torch.nn as nn 
import torch 
from quant.quant_layer import UniformAffineQuantizer
import torch.nn.functional as F

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
        self.factor = 2 ** (8 - n_bits) 

        act_scale_refactor = act_scale / self.factor
        self.register_buffer("act_zp", torch.tensor(act_zp, dtype=torch.int32))
        self.register_buffer("act_scale", torch.tensor(act_scale_refactor, dtype=torch.float32))

        w_scales_refactor = w_scales.detach().float() / self.factor 
        self.register_buffer("w_zps", w_zps.to(torch.int32))
        self.register_buffer("w_scales", w_scales_refactor.to(torch.float32))

        # float weight/bias 보관 
        self.register_buffer("weight_fp", conv.weight.detach().float())
        if conv.bias is not None:
            self.register_buffer("bias_fp", conv.bias.detach().float())
            self.bias_fp = self.bias_fp 
        else:
            self.bias_fp = None 

    def forward(self, x): 
       # 1) activation Q -> DQ (per-tensor, quint8)
        x_dq = torch.fake_quantize_per_tensor_affine(
            x,
            self.act_scale,
            self.act_zp,           
            quant_min=-128,
            quant_max=127
        )
        # x_dq = torch.dequantize(x_q)

        # 2) wegiht Q -> DQ (per-channel, qint8) 
        w_dq = torch.fake_quantize_per_channel_affine(
            self.weight_fp,
            self.w_scales,
            self.w_zps,
            self.axis,
            quant_min=-128,
            quant_max=127
        )


        y = F.conv1d(
            x_dq, w_dq, self.bias_fp, stride = self.conv.stride, padding = self.conv.padding
        )
        return y 
    
class QDQLinear(nn.Module): 
    def __init__(self, linear: nn.Linear,
                 act_scale: float,
                 act_zp: int, 
                 w_scales: torch.Tensor,
                 w_zps: torch.Tensor,
                 axis: int = 0,
                 n_bits: int = 8): 
        
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.linear = linear 
        out_features, in_features = linear.weight.shape
        self.factor = 2 ** (8 - n_bits)

        act_scale_refactor = act_scale / self.factor 
        self.register_buffer("act_scale", torch.tensor(act_scale_refactor, dtype=torch.float32))
        self.register_buffer("act_zp", torch.tensor(act_zp, dtype=torch.int32))

        # weight quant (per-channel, qint8 on axis=0 -> out_features)
        assert w_scales.numel() == out_features
        assert w_zps.numel()    == out_features

        w_scales_refactor = w_scales.detach().float().view(-1) / self.factor
        self.register_buffer("w_scales", w_scales_refactor.to(torch.float32))
        self.register_buffer("w_zps",    w_zps.to(torch.int32))
        self.axis = axis  # Linear/Conv는 보통 0 (out_features)

        # 원본 FP 가중치/바이어스 저장(ONNX에서 initializer로 잡히게)
        self.register_buffer("weight_fp", linear.weight.detach().float())
        if linear.bias is not None:
            self.register_buffer("bias_fp", linear.bias.detach().float())
        else:
            self.bias_fp = None

    def forward(self, x):
        # Linear 입력은 (..., in_features) 어떤 rank도 가능. 일단 FP32로
        # x = x.float()

        # 1) activation Q -> DQ (per-tensor, quint8)
        x_dq = torch.fake_quantize_per_tensor_affine(
            x,
            self.act_scale,
            0,
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
        # 3) float linear (ONNX에선 DQ -> Gemm/MatMul(+Add))
        y = F.linear(x_dq, w_dq, self.bias_fp)
        return y


class QDQConv1d_TaDA(nn.Module):
    def __init__(self, conv: nn.Conv1d,
                 w_scales: torch.Tensor,
                 weight_quantizer: UniformAffineQuantizer,
                 set_8bit: bool = True,
                 act_scale: float = None):
        
        super().__init__()
        assert isinstance(conv, nn.Conv1d)
        self.conv = conv
        self.weight_quantizer = weight_quantizer
        self.set_8bit = set_8bit
        self.axis = 0  # out_features

        self.register_buffer("act_scale", torch.tensor(act_scale, dtype=torch.float32))
        self.register_buffer("act_zp", torch.tensor(0, dtype=torch.int32))

        self.register_buffer("w_scales", w_scales.to(torch.float32))
        self.register_buffer("w_zps", torch.zeros_like(self.w_scales, dtype=torch.int32))

    def forward(self, x):
        # 1) activation quant
        if self.set_8bit: # quantmodule 
            x_q = torch.fake_quantize_per_tensor_affine(
            x,
            self.act_scale, 
            self.act_zp, # zp 
            quant_min=-128,
            quant_max=127
            )
        else: # custom quantmodule 
            x_q = x

        if self.set_8bit: # quantmodule 
            w_q = torch.fake_quantize_per_channel_affine(
            self.conv.weight, 
            self.w_scales, 
            torch.zeros_like(self.w_scales, dtype = torch.int32),
            self.axis,
            quant_min = -128,
            quant_max = 127
        )
        else: # custom quantmodule 
            w_q = self.weight_quantizer(self.conv.weight)

        b = self.conv.bias

        y = F.conv1d(
            x_q, w_q, b,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        return y

class QDQLinear_TaDA(nn.Module): 
    def __init__(self, linear: nn.Linear,
                 w_scales: torch.Tensor,
                 act_scale: float = None,
                 set_8bit: bool = True,
                 n_bits: int = 8):   # 원래 QNN 비트
        
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.linear = linear
        self.set_8bit = set_8bit 
        self.axis = 0  # out_features
        self.factor = 2 ** (8 - n_bits)  # 16

        act_scale_refactor = act_scale / self.factor 
        self.register_buffer("act_scale", torch.tensor(act_scale_refactor, dtype=torch.float32))
        self.register_buffer("act_zp", torch.tensor(0, dtype=torch.int32))

        # weight: per-channel [Cout]
        w_scales_refactor = w_scales.detach().float().view(-1) / self.factor
        self.register_buffer("w_scales", w_scales_refactor.to(torch.float32))
        self.register_buffer("w_zps", torch.zeros_like(self.w_scales, dtype=torch.int32))


        # float weight/bias
        self.register_buffer("weight_fp", linear.weight.detach())
        if linear.bias is not None:
            self.register_buffer("bias_fp", linear.bias.detach())
        else:
            self.bias_fp = None

    def forward(self, x):
        # 1) activation QDQ (per-tensor, INT8)
        if self.set_8bit:
            x_dq = torch.fake_quantize_per_tensor_affine(
                x,
                self.act_scale,
                self.act_zp,
                quant_min=-128,
                quant_max=127
            )
        else:
            x_dq = x

        # 2) weight QDQ (per-channel, INT8)
        w_dq = torch.fake_quantize_per_channel_affine(
            self.weight_fp,
            self.w_scales,
            self.w_zps,
            self.axis,
            quant_min=-8,
            quant_max=7,
        )

        # 3) float matmul
        y = F.linear(x_dq, w_dq, self.bias_fp)
        
        return y

