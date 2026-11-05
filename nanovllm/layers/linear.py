import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    # TP 切分专用除法：必须整除，否则静默截断会引发难以排查的错误，故断言
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    # 各类并行线性层的公共基类：持有完整形状的权重元信息 + 权重加载钩子

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        # 沿哪个维度切分：0 = 输出维（列并行），1 = 输入维（行并行），None = 不切
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 权重按“本 rank 应持有的分片形状”创建（子类传入的 size 已是切分后的）
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 把加载函数挂到参数对象上：loader 加载权重时发现参数带 weight_loader
        # 就调用它完成 TP 切分拷贝，而不是整块复制（函数挂属性 = 轻量策略模式）
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            # 显式注册 None：nn.Module 语义上“有 bias 字段但为 None”，
            # state_dict 与 F.linear(x, w, None) 都能正确处理
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基类不实现前向：由列/行并行的子类决定是否通信
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    # 不切分的线性层：每个 rank 持有完整权重（本仓库目前没有使用方，保留作对照）

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 整块直接拷贝
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 普通 matmul，无通信
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    # 列并行：把输出维切成 tp_size 份，每个 rank 算自己的输出切片。
    # 输入 x 各 rank 完整相同，输出各持一块（通常交给下游的行并行层消化，无需通信）

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 本 rank 权重形状 = [output_size/tp, input_size]；tp_dim=0 表示按输出维切
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # 本 rank 分片在完整权重里占的行数
        shard_size = param_data.size(self.tp_dim)
        # 第 rank 个切片的起始行
        start_idx = self.tp_rank * shard_size
        # narrow(dim, start, length)：零拷贝视图取出 [start, start+shard) 区间再拷入
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每个 rank 对同一输入算不同的输出切片
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    # “合并式”列并行：把多个同形权重（如 gate_proj + up_proj）合并进一个大层，
    # 一次 GEMM 同时算两支，减少 kernel 启动次数

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        # 记录每个子层在输出维上占的大小（切分前），weight_loader 按它定位偏移
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        # 子层在大权重里的起始偏移：前面各子层大小之和，再按 TP 折算成本 rank 分片的行数
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        # 本子层在本 rank 中的行数
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 先在大参数里定位本子层的区间
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 完整权重按 TP 份数切块，取本 rank 那块拷入
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    # QKV 合并列并行：Q/K/V 三种投影拼成一个权重矩阵（一次 GEMM 出 QKV）。
    # GQA 下 K/V 头数少于 Q 头数，三者在输出维上的宽度不同，加载逻辑也各自不同

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 未提供 KV 头数则视为 MHA（KV 头数 = Q 头数）
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        # 本 rank 分到的 Q 头数与 KV 头数（各自必须能被 tp_size 整除）
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 合并后的输出宽度：Q 头 + 2 份 KV 头（K 一份、V 一份）
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            # Q 段：宽度 = 本 rank 的 Q 头数 * head_dim，位于输出维最前
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            # K 段：宽度按 KV 头数算，紧跟在 Q 段之后
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            # V 段：与 K 等宽，排在 K 段之后
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        # 在本 rank 的合并参数里定位目标段
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # HF 权重按 TP 份数切块取本 rank 的
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    # 行并行：把输入维切成 tp_size 份，每个 rank 用完整输出的 1/tp_size 做部分和，
    # 再 all_reduce 求和。常接在列并行之后：x 已经是各 rank 一份切片，天然衔接

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 本 rank 权重形状 = [output_size, input_size/tp]；tp_dim=1 表示按输入维切
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            # 一维张量是 bias：bias 不切分（只在 rank 0 加，见 forward），整块拷贝
            param_data.copy_(loaded_weight)
            return
        # 二维权重沿输入维取本 rank 的列切片
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 各 rank 算出“部分和”：只覆盖输入的一部分维度，结果不完整
        # bias 只加一次（rank 0），否则 all_reduce 后会被加 tp_size 遍
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            # all_reduce 求和：把各 rank 的部分和累加成完整输出，所有 rank 得到相同结果
            dist.all_reduce(y)
        return y
