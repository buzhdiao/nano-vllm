import torch
from torch import nn


class RMSNorm(nn.Module):
    # RMSNorm：相比 LayerNorm 去掉均值中心化，只做“均方根缩放”，计算量更小。
    # Qwen3 全系使用 RMSNorm，且带残差融合的 add_rms_forward（QK-Norm 也复用它）

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        # 可学习的逐通道缩放向量，初始化为全 1（等价于恒等缩放）
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # 记录原始 dtype（通常是 bf16），计算在 fp32 里做防溢出/精度损失
        orig_dtype = x.dtype
        x = x.float()
        # 每个位置的均方值：沿最后一维（特征维）求平方均值
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # 原位乘以 1/sqrt(var+eps)：RMSNorm 的归一化核心
        x.mul_(torch.rsqrt(var + self.eps))
        # 转回原 dtype 后乘可学习权重（缩放放在 dtype 转换后，与 HF 实现数值一致）
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 融合版“残差相加 + RMSNorm”：把本层输出 x 加回残差流，
        # 并把和同时作为 (a) 本层归一化的输入 (b) 传给下一层的残差。
        # 融合后少一次显存读写往返，是 decoder 段最大的免费性能来源之一
        orig_dtype = x.dtype
        # fp32 下原位相加，避免 bf16 溢出
        x = x.float().add_(residual.float())
        # 更新后的残差流（转回原 dtype 存储，省一半显存带宽）
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        # 返回 (归一化输出, 新残差)：调用方两个都接着用
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 无残差输入（第一层/embedding 后）走普通路径，否则走融合路径
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
