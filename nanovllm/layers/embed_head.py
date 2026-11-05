import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    # 词表并行 embedding：把词表维切成 tp_size 份，每个 rank 只持有一段词向量。
    # 查表后用 all_reduce 合成完整结果（vLLM 同名实现的精简版）

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 词表必须能被 TP 均分
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        # 本 rank 持有的词表段大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 本 rank 负责的词表 id 区间 [start, end)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # 沿词表维（第 0 维）取本 rank 的那段词向量
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 布尔掩码：标记哪些 token 的 id 落在本 rank 的词表段内
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 把全局 id 平移成本地 id（段内从 0 计）；
            # 落段外的 id 平移后为 0（会查到任意行，稍后被掩码清零）
            x = mask * (x - self.vocab_start_idx)
        # 本地查表
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 段外位置的结果清零（False 按 0 参与乘法）
            y = mask.unsqueeze(1) * y
            # all_reduce 求和：每个 token 只有一个 rank 查到了非零向量，求和即完整 embedding
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    # 输出头与 embedding 共用同一实现：权重形状同为 [vocab, hidden]，
    # 差别只在 forward——这里是线性投影（hidden -> vocab 分片），不是查表

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        # LM head 不支持 bias（与 HF Qwen3 一致）
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # prefill 批是“扁平拼接的多序列”：只有每条序列最后一个 token 的
            # hidden state 需要 logits。cu_seqlens_q[1:]-1 即每条序列末 token 的下标
            last_indices = context.cu_seqlens_q[1:] - 1
            # 高级索引取出这些行；contiguous() 消除高级索引产生的跨步布局
            x = x[last_indices].contiguous()
        # 与本地词表分片做线性投影：得到的是“本 rank 那段词表”的 logits
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # gather 把各 rank 的词表段 logits 聚到 rank 0，再沿词表维拼接成完整 logits。
            # 只有 rank 0 需要完整 logits（采样只在 rank 0），其余 rank 传 None
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
