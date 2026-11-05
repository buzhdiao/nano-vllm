import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    # 单层自注意力模块：QKV 投影 -> (可选 QK-Norm) -> RoPE -> 分页注意力 -> 输出投影

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        # Q 头数必须能被 TP 均分
        assert self.total_num_heads % tp_size == 0
        # 本 rank 分到的 Q 头数
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        # KV 头数（GQA 时少于 Q 头数）同样必须能被 TP 均分
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        # 每头维度：优先用配置显式值，否则按 hidden/heads 推出
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # 本 rank 的 Q 宽度与 KV 宽度（split QKV 时按这两个尺寸切）
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # softmax 缩放因子 1/sqrt(d)
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # Q/K/V 合并成一个列并行层：一次 GEMM 同时产出三支（K/V 按 GQA 头数较窄）
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 输出投影用行并行：输入（注意力输出）各 rank 已是切片，算完 all_reduce
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            # 部分 Qwen3 版本把 rope_theta 写在 rope_scaling 里，这里兼容读取
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        # RoPE 模块：get_rope 内部 lru_cache，全模型各层共享同一实例
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        # 纯注意力核（无投影）：负责写 KV cache 和调 flash_attn
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            # Qwen3 特有的 QK-Norm：对每个头的 Q/K 各做一次 RMSNorm（在 RoPE 之前），
            # 用于稳定注意力 logits 的量级
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 一次 GEMM 算出本 rank 的 QKV（输出维上 Q|K|V 三段并排）
        qkv = self.qkv_proj(hidden_states)
        # 按各自宽度切成三段
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 重塑成 flash_attn 要求的 [tokens, heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            # QK-Norm：逐头归一化
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 施加旋转位置编码（Q/K 同步旋转）
        q, k = self.rotary_emb(positions, q, k)
        # 写 KV cache + 注意力计算（prefill/decode 分派在层内完成）
        o = self.attn(q, k, v)
        # 压回头维成 [tokens, heads*dim] 再做输出投影（行并行 + all_reduce）
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    # SwiGLU 前馈：gate_up 合并列并行 -> SiluAndMul -> down 行并行

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # gate_proj 与 up_proj 合并成一个列并行大层，一次 GEMM 出两支
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # 降维投影行并行：结尾 all_reduce 汇总各 rank 的部分和
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        # 本实现只支持 silu 激活（Qwen3 即如此）
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # 升维：一次 GEMM 得 [gate 支路 | up 支路]
        gate_up = self.gate_up_proj(x)
        # 门控激活：silu(gate) * up
        x = self.act_fn(gate_up)
        # 降维回 hidden_size
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    # 标准 decoder 层：RMSNorm -> 注意力 -> 残差 -> RMSNorm -> MLP -> 残差

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            # getattr 三连：容忍不同版本 Qwen3Config 的可选字段缺省
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 注意力前与 MLP 前各一个 RMSNorm（Pre-Norm 结构）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # 序列开头（embedding 输出）：residual 尚未建立。
            # 残差流取“归一化前的输入”，即 hidden 本身；注意这里归一化与残差拆开
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 常规路径：融合的“加残差 + 归一化”，一次读写完成两件事
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        # 注意力输出并入残差流并归一化，供 MLP 使用
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        # 返回 (本层输出, 最新残差流)：残差真正累加发生在下一层的 add_rms_norm 里
        return hidden_states, residual


class Qwen3Model(nn.Module):
    # 去 lm_head 的主干：embedding + N 层 decoder + 最终 RMSNorm

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 词表并行 embedding
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # N 层 decoder
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 查词向量
        hidden_states = self.embed_tokens(input_ids)
        # 残差流从 None 起步（第一层里建立）
        residual = None
        for layer in self.layers:
            # 每层返回 (新输出, 新残差)，残差逐层向后传递
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后一层：把残差并入并做最终归一化（丢弃返回的残差）
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    # 顶层封装：主干 + LM head；packed_modules_mapping 告诉权重加载器
    # “HF 里的独立投影应合并加载到本模型的哪个大层的哪个分片”
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        # 输出头也是词表并行的（与 embedding 同一套切分）
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            # 小模型（如 0.6B）常做权重绑定：直接共享同一块存储（.data 赋值共享底层张量），
            # 加载时 loader 会往两个名字写同一份内存
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 只跑到主干最后的 hidden states；logits 单独由 compute_logits 计算
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # hidden -> logits（内含 prefill 末 token 挑选与 TP gather）
        return self.lm_head(hidden_states)
