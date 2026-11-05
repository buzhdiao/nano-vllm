import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# Triton JIT 内核：把本步新算出的 K/V 写进分页 KV cache。
# 每个 CUDA 程序（program）负责一个 token 的整行 K/V 拷贝
@triton.jit
def store_kvcache_kernel(
    key_ptr,          # 新算出 K 的首地址（扁平 [N, num_heads*head_dim] 视图）
    key_stride,       # K 第 0 维步长：相邻两个 token 的起始地址差
    value_ptr,        # 新算出 V 的首地址
    value_stride,     # V 第 0 维步长
    k_cache_ptr,      # KV cache 中本层 K 的首地址（[num_blocks*block_size, D] 视图）
    v_cache_ptr,      # 本层 V 的首地址
    slot_mapping_ptr, # 每个 token 的目标 slot（cache 内一维下标），-1 表示不写
    D: tl.constexpr,  # 每行元素数 = num_heads * head_dim，constexpr 让编译器展开循环
):
    # 当前 program 处理第 idx 个 token（grid 一维并行，N 个程序）
    idx = tl.program_id(0)
    # 读该 token 的目标 slot
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    # CUDA Graph 回放时桶内多余行 slot=-1，直接返回，避免写坏 cache
    # 计算 K 中该 token 行的元素偏移：行起点 idx*stride，行内 0..D-1 连续
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 从新 K/V 里读出这一行
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # cache 按 [slot, D] 排布：目标偏移 = slot*D + 行内偏移，
    # “散落各处的 token”由此写到“任意物理块的任意格子”
    cache_offsets = slot * D + tl.arange(0, D)
    # 写 K 与 V 到 cache
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    # Python 侧薄封装：校验内存布局假设后启动 kernel
    N, num_heads, head_dim = key.shape
    # 把最后两维折成一维行宽：kernel 只面向 [N, D] 的扁平视图工作
    D = num_heads * head_dim
    # 以下断言保证“按 stride 手工寻址”的合法性：
    # 最后一维必须连续（stride==1），否则 tl.arange(0,D) 的连续偏移不成立
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    # 头与头之间也必须紧挨（没有 padding）：stride(1)==head_dim
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # cache 视图的行宽同样是 D（即 cache 最后一维恰为 num_kv_heads*head_dim 连续存放）
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    # token 数与 slot 数一致：一 token 一槽
    assert slot_mapping.numel() == N
    # 启动 N 个 program 的一维 grid
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    # 通用注意力层：不含投影（投影在模型的 QKVParallelLinear/o_proj 里），
    # 只负责“写 cache + 按上下文选 flash_attn 入口”

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # softmax 缩放因子 = 1/sqrt(head_dim)
        self.scale = scale
        # KV 头数可与 Q 头数不同（GQA/MQA）
        self.num_kv_heads = num_kv_heads
        # 占位空张量：分配 KV cache 后由 ModelRunner 换成真正的 cache 切片。
        # 用 numel()==0 判断“cache 是否已就绪”
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 从全局上下文拿本步元数据（slot_mapping、cu_seqlens、block_tables 等）
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            # 第一步：先把新 K/V 写进分页 cache（prefill/decode 都要写）
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                # 命中 prefix cache：Q 是新算的，但 K/V 要用 cache 里的完整历史
                # （含之前命中的块），所以把 k/v 指向整块 cache，
                # 由 block_table + cu_seqlens_k 界定每条序列实际取多长
                k, v = k_cache, v_cache
            # varlen 接口：批内各序列长度不一，扁平拼接 + cu_seqlens 切分；
            # causal=True 保证只看前文；block_table 传给 paged KV 读取
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # decode 接口：q 形状 [bs, 1, heads, dim]（unsqueeze 出 seq 维=1），
            # K/V 直接给整块 cache，按 context_lens（每条可见长度）+ block_table 取页
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
