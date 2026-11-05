import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    # 默认加载策略：整块拷入（形状由 HF 权重与参数天然一致保证）
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    # 从 HF safetensors 目录加载权重到（可能已 TP 切分的）模型。
    # 特点：零形状转换——参数在建模时已按 TP 切小，这里逐权重“找主人 + 调钩子”
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    # 遍历目录下所有 safetensors 分片文件（大模型常拆成多个文件）
    for file in glob(os.path.join(path, "*.safetensors")):
        # safe_open 惰性打开：meta 只读头部，get_tensor 时才真正读数据，省内存
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 命中合并映射（如 model.layers.0.self_attn.q_proj.weight）：
                        # 换名找到合并后的参数（qkv_proj），并带上分片 id（"q"）
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        # 合并参数必带 weight_loader 钩子：由它完成“定位分片 + 拷贝”
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # for-else：内层循环没有 break（未命中任何映射）= 普通一一对应权重。
                    # get_parameter 按点分名逐级取子模块参数
                    param = model.get_parameter(weight_name)
                    # 参数若自带 weight_loader（如 TP 切分层）用之，否则用默认整块拷贝
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
