from dataclasses import dataclass


# 采样参数：dataclass(slots=True) 自动生成 __init__，字段即超参，带默认值方便快速构造
@dataclass(slots=True)
class SamplingParams:
    # 温度系数：logits 先除以 temperature 再 softmax；越大越随机，趋近 0 接近贪心
    temperature: float = 1.0
    # 最多生成的 token 数（不含 prompt），达到即强制结束
    max_tokens: int = 64
    # True 则忽略 EOS token 继续生成（benchmark 里用来测满固定输出长度）
    ignore_eos: bool = False

    # dataclass 的初始化后钩子，用于参数校验
    def __post_init__(self):
        # 温度必须为正：本引擎的采样实现依赖除以 temperature，
        # temperature=0（贪心解码）会触发除零，故显式拒绝并提示
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
