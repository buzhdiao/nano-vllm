# 06 - 注意力层与 flash_attn 集成（varlen / with_kvcache）

`layers/attention.py`（108 行）的 `Attention` 是一个"无投影"的纯注意力核适配层：Q/K/V 的线性投影在模型的 `QKVParallelLinear`/`o_proj` 里，这一层只做两件事——**把新 K/V 写进分页 cache**，以及**按批次形态选 flash_attn 的两个入口之一**。

## 两个 flash_attn 入口

推理引擎的难点在于：prefill 批内各序列长度不一，decode 时每序列又只有 1 个新 token 却要读全部历史 KV。flash_attn 为这两种形态提供专门接口：

**prefill：`flash_attn_varlen_func`**（attention.py:98-102）。变长批不用 padding 补齐，而是把所有 token **扁平拼接**成 `[Σtokens, heads, dim]`，用 `cu_seqlens_q/cu_seqlens_k`（累积前缀和，int32）告诉 kernel 每条序列的边界，`max_seqlen_q/k` 供 kernel 调度。`causal=True` 保证只看前文。这些元数据由 `ModelRunner.prepare_prefill` 打包（model_runner.py:180-241）经全局 Context 传入（utils/context.py）。

**decode：`flash_attn_with_kvcache`**（attention.py:105-108）。q 形状 `[bs, 1, heads, dim]`（`unsqueeze(1)` 补出 seq 维），K/V 直接给**整块 cache 张量**，kernel 按 `cache_seqlens`（每序列可见长度）和 `block_table`（页表）在 cache 里跳跃取页——这就是 flash_attn 内建的 PagedAttention 支持。

## prefix cache 命中时的分支

命中缓存时情况特殊（attention.py:93-95）：Q 只有本步新算的部分，但 K/V 必须覆盖"缓存块 + 新块"的完整历史。此时直接把 `k, v = k_cache, v_cache` 指向 cache 本体，配合 `block_table` 与更长的 `cu_seqlens_k`，让 kernel 自己去 cache 里按页取数。未命中时（首轮 prefill）K/V 还没进 cache（写入发生在同一步的 store_kvcache），直接用刚算出的 k/v 张量即可。

注意顺序：`store_kvcache` 先写 cache（attention.py:87-89），随后 attention 读取——同一 stream 内写读有序。

## KV cache 的布局

cache 由 ModelRunner 一次性预分配为 `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`（model_runner.py:161），第 0 维 0 是 K、1 是 V，按层切片挂到各 Attention 模块（model_runner.py:167-170）。每个 Attention 层看到的 `k_cache` 是 `[num_blocks, block_size, num_kv_heads, head_dim]` 的连续视图，store kernel 把它当作 `[num_blocks*block_size, D]` 的二维矩阵按 slot 一维寻址（见 07 篇）。

## GQA 与缩放

Qwen3 用 GQA：Q 头多、KV 头少。flash_attn 原生支持 `heads_q != heads_kv`（按组广播）；本 rank 的头数在 `Qwen3Attention` 里算好传入（qwen3.py:33-40）。`softmax_scale = 1/sqrt(head_dim)`（qwen3.py:39）。

## torch.compile 的角色

RoPE、RMSNorm、激活、采样都套了 `@torch.compile`（如 rotary_embedding.py:50、layernorm.py:19/35），把小算子融合成大 kernel。唯独 Attention 本体不编译——flash_attn 是外部算子，编译收益小且变长形状易触发重编译。

整体链路：`QKVParallelLinear` → split → (QK-Norm) → RoPE → `Attention.forward`（写 cache + flash_attn）→ `o_proj`，见 qwen3.py:86-108。
