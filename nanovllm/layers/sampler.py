import torch
from torch import nn


# 采样器：温度缩放 + 多项式采样，全部在一个 compiled kernel 里完成
class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 温度缩放：原位除以各序列自己的温度（unsqueeze 成 [bs,1] 以广播）；
        # 先转 float32 保证精度（logits 可能是 bf16）
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # softmax 得到概率分布
        probs = torch.softmax(logits, dim=-1)
        # Gumbel/指数竞赛技巧：每个概率除以独立的 Exp(1) 噪声后取 argmax，
        # 数学上等价于按 probs 多项式采样一次——避免了昂贵的 torch.multinomial 及其同步
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
