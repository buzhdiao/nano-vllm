import torch
from torch import nn
import torch.nn.functional as F


# SwiGLU 的激活半部：输入前半是 gate（过 SiLU），后半是 value（直通），逐元素相乘。
# 与 nn.Module 组合即可被 torch.compile 整体捕获
class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 最后一维对半切开：x 为 gate 支路，y 为 value 支路（Qwen3 的 gate_up_proj 输出如此排布）
        x, y = x.chunk(2, -1)
        # silu(x) = x * sigmoid(x)：门控放行比例由 x 自己决定，再与 value 相乘
        return F.silu(x) * y
