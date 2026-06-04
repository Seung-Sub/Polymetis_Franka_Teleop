import torch
import torch.nn as nn
from quant.quant_layer import QuantModule, QuantModuleLoRA, QuantModuleLoRATaDA, StraightThrough, UniformAffineQuantizer

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
                if m.act_quantizer is None:   
                    m.act_quantizer = UniformAffineQuantizer(**act_params, need_init=True)
                    m.use_act_quant = True


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, QuantModule):  ## remove BaseQuantBlock
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def set_first_last_layer_to_8bit(self):
    
        name2mod = {n:m for n,m in self.model.named_modules() if isinstance(m, QuantModule)}

        # time embeding 2 layers + first conv layer + final conv layer 

        for key in ["net.diffusion_step_encoder.1", "net.diffusion_step_encoder.3", "net.down_modules.0.0.blocks.0.block.0", "net.final_conv.1"]:
            if key in name2mod: 
                name2mod[key].weight_quantizer.bitwidth_refactor(8)
                name2mod[key].act_quantizer.bitwidth_refactor(8)
                name2mod[key].ignore_reconstruction = True 

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
            if isinstance(m, (QuantModuleLoRA, QuantModule)):  ## remove BaseQuantBlock
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, t, global_cond=None, **kwargs):
        out = self.model(x, t, global_cond=global_cond, **kwargs)
        return out
    
    def set_first_last_layer_to_8bit(self):

        name2mod = {n:m for n,m in self.model.named_modules() if isinstance(m, (QuantModuleLoRA, QuantModule))}

        # time embeding 2 layers + first conv layer + final conv layer 
        for key in ["net.diffusion_step_encoder.1", "net.diffusion_step_encoder.3", "net.down_modules.0.0.blocks.0.block.0", "net.final_conv.1"]:
            if key in name2mod: 
                print("[INFO] set 8bit for ", key)
                name2mod[key].weight_quantizer.bitwidth_refactor(8)
                name2mod[key].act_quantizer.bitwidth_refactor(8)
                name2mod[key].ignore_reconstruction = True 

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, (QuantModuleLoRA, QuantModule)):
                module_list += [m]
        module_list[-1].disable_act_quant = True

class QuantModelLoRATaDA(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, num_steps=100):
        super().__init__()
        # search_fold_and_remove_bn(model)
        self.model = model
        self.num_steps = num_steps
        self._t_norm = None       
        self.count = 0
        self.total_count = 47 ## 47 for DiffusionPolicy-C
        self.special_module_count_list = [7,8,23,47] ## fist, last layer
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

            if isinstance(child_module, (nn.Conv1d, nn.Linear)) and 'skip' not in name and 'op' not in name:
                self.count += 1
                if self.count in self.special_module_count_list:
                    qm = QuantModule(child_module, weight_quant_params, act_quant_params, need_init=True)
                else:
                    qm = QuantModuleLoRATaDA(child_module, weight_quant_params, act_quant_params)

                import weakref 
                object.__setattr__(qm, "t_provider_ref", weakref.ref(self))
                setattr(module, name, qm)

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModuleLoRATaDA, QuantModule)):  ## remove BaseQuantBlock
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, t, global_cond=None, **kwargs): 
        t = t.to(dtype=torch.long).reshape(-1)                          
        den = (self.num_steps - 1) if (self.num_steps > 1) else 1
        self._t_norm = (t.to(torch.float32) / float(den)).unsqueeze(1)  # [B,1]
        out = self.model(x, t, global_cond=global_cond)
        self._t_norm = None
        return out
    
    def set_first_last_layer_to_8bit(self):

        name2mod = {n:m for n,m in self.model.named_modules() if isinstance(m, (QuantModuleLoRATaDA, QuantModule))}

        # time embeding 2 layers + first conv layer + final conv layer 
        for key in ["net.diffusion_step_encoder.1", "net.diffusion_step_encoder.3", "net.down_modules.0.0.blocks.0.block.0", "net.final_conv.1"]:
            if key in name2mod: 
                print("[INFO] set 8bit for ", key)
                name2mod[key].weight_quantizer.bitwidth_refactor(8)
                name2mod[key].act_quantizer.bitwidth_refactor(8)
                name2mod[key].ignore_reconstruction = True 

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, (QuantModuleLoRATaDA, QuantModule)):
                module_list += [m]
        module_list[-1].disable_act_quant = True

