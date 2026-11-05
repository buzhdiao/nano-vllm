from dataclasses import dataclass
import torch


# 前向上下文：一次 forward 所需的全部“形状/元数据”打包。
# 好处：attention 层、lm_head 等深层模块不必层层透传十几个参数，
# ModelRunner 装包、需要的层自取（vLLM ForwardContext 的同思路精简版）。
# slots=True：实例字段固定，省内存且访问更快；每步都新建实例，GC 压力小
@dataclass(slots=True)
class Context:
    # 本步是 prefill（多 token 拼接）还是 decode（每序列 1 token）
    is_prefill: bool = False
    # prefill 用：Q/K 的累积序列长度（flash_attn varlen 接口）
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    # prefill 用：批内最长 Q/K 序列长度（kernel 调度的上界）
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # 本步每个新 token 在 KV cache 中的写入位置（一维 slot 下标）
    slot_mapping: torch.Tensor | None = None
    # decode 用：每条序列当前可见的上下文长度
    context_lens: torch.Tensor | None = None
    # 分页寻址表：[batch, max_blocks]，-1 表示无效页
    block_tables: torch.Tensor | None = None

# 模块级单例：全局唯一上下文实例（进程内共享，各 TP 进程各自持有一份）
_CONTEXT = Context()

def get_context():
    # 读取当前上下文：attention/lm_head 的 forward 在计算中调用
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    # 整体替换为新实例：一次 forward 一套元数据，切换成本低且无并发写问题
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    # 用后复位成空上下文：防止下一步（如 warmup、图捕获）误读上一步的元数据
    global _CONTEXT
    _CONTEXT = Context()
