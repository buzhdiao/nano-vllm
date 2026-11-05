import os
from dataclasses import dataclass
from transformers import AutoConfig


# @dataclass(slots=True)：dataclass 自动生成 __init__/__repr__；
# slots=True 为每个实例生成 __slots__，省去 per-instance 的 __dict__，
# 属性访问更快、内存更省（Config 全局只有一份，收益有限，主要是统一风格）。
@dataclass(slots=True)
class Config:
    # 模型本地目录路径（必须含 config.json 与 safetensors 权重），唯一必填项
    model: str
    # 单次 prefill 批内最多一起计算的 token 数；既限制显存峰值，也是 chunked prefill 的切分粒度
    max_num_batched_tokens: int = 16384
    # 引擎同一时刻最多并行维持的序列数（decode 批大小的上限）
    max_num_seqs: int = 512
    # 单条序列允许的最大总长度（prompt + 生成）；会与模型自身 max_position_embeddings 取 min
    max_model_len: int = 4096
    # 允许 KV cache 占用的显存比例（0-1），决定能分多少个 block、能跑多长上下文
    gpu_memory_utilization: float = 0.9
    # 张量并行度：=1 单卡；>1 时为每个额外 rank 起一个子进程
    tensor_parallel_size: int = 1
    # True 则禁用 CUDA Graph，全部走 eager 执行（调试用，或第一次跑通时排除 graph 问题）
    enforce_eager: bool = False
    # 从模型目录读出的 HuggingFace 配置（__post_init__ 里填充），暴露给各层使用
    hf_config: AutoConfig | None = None
    # 结束符 token id，初始 -1 表示“未设置”，由 LLMEngine 用 tokenizer 的 eos 覆写
    eos: int = -1
    # KV cache 的分页块大小（每个 block 存多少个 token 的 K/V）
    kvcache_block_size: int = 256
    # KV cache 的 block 总数，-1 表示“未计算”，由 ModelRunner 按剩余显存算出后回填
    num_kvcache_blocks: int = -1

    # dataclass 生成 __init__ 后会自动调用的钩子，用来做校验和派生字段的初始化
    def __post_init__(self):
        # 模型路径必须是本地已存在的目录（本引擎不做下载）
        assert os.path.isdir(self.model)
        # block 大小必须是 256 的倍数：与 flash_attn 的 page 尺寸约定对齐，也便于哈希对齐
        assert self.kvcache_block_size % 256 == 0
        # 限制 TP 规模在 1~8 卡之间（单机多卡场景）
        assert 1 <= self.tensor_parallel_size <= 8
        # 读取 HuggingFace 格式的模型配置（hidden_size、层数、头数、dtype 等都在这里）
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 用户配置的长度不能超过模型训练时支持的最大位置数，取二者较小值兜底
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
