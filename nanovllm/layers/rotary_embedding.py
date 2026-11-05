from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # RoPE 的“前后对半”实现（GPT-NeoX 风格）：把头维劈成两半做二维旋转，
    # 而不是 HF 早期 rotate_half 的交错劈法——效果等价，cache 排布更简单
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 二维旋转矩阵作用于 (x1, x2)：y1 = x1*cos - x2*sin
    y1 = x1 * cos - x2 * sin
    # y2 = x2*cos + x1*sin
    y2 = x2 * cos + x1 * sin
    # 拼回完整头维并转回原 dtype（旋转在 fp32 中计算）
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        # 本实现只支持“整头都做旋转”（rotary_dim == head_size）
        assert rotary_dim == head_size
        # 逆频率：base^(-2i/d)，i 为频率对下标。低维转得快（高频），高维转得慢（低频）
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 所有绝对位置 0..max_pos-1
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 外积：[max_pos, d/2] 的相位表，einsum 一步算出位置 x 频率网格
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # 预计算 cos/sin 缓存拼成一张表：[max_pos, d]（前半 cos 后半 sin），
        # unsqueeze_(1) 增加头维便于广播到 [pos, 1, d]
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # register_buffer：成为模块状态但不是参数（不参与梯度/优化器）；
        # persistent=False 表示不写入 state_dict——缓存可由 base/维数随时重建
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 按每个 token 的绝对位置查 cos/sin（decode 时 positions 是每序列一个下标）
        cos_sin = self.cos_sin_cache[positions]
        # 表里前半是 cos 后半是 sin：切出来
        cos, sin = cos_sin.chunk(2, dim=-1)
        # Q 与 K 做同样的旋转（V 不旋转）
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    # 工厂 + 进程级缓存：同参数的 RoPE 模块全模型只建一次
    # （每层 Attention 都会调用，参数相同，lru_cache(1) 直接复用同一实例）
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
