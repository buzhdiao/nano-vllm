from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


# 序列的三种生命周期状态：等待调度 -> 运行中 -> 已结束
class SequenceStatus(Enum):
    WAITING = auto()   # 在 waiting 队列中，尚未获得 KV cache block
    RUNNING = auto()   # 已分配 block，正在逐步 decode
    FINISHED = auto()  # 命中 EOS 或达到 max_tokens


class Sequence:
    # 类属性：每个 KV block 能存多少 token（LLMEngine 启动时用 config 覆写）。
    # 放在类上而不是实例上，是因为 BlockManager 和调度器也要读它，且 pickle 传输时不会序列化类属性
    block_size = 256
    # 类属性的全局自增计数器：itertools.count 返回无限迭代器，
    # next() 取号保证每条序列有全局唯一 seq_id（输出按 seq_id 排序还原请求顺序）
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 领取全局唯一 id；注意用类属性 Sequence.counter 而非 self.counter，所有实例共享同一个计数器
        self.seq_id = next(Sequence.counter)
        # 新序列从 WAITING 状态起步，等调度器分配 block
        self.status = SequenceStatus.WAITING
        # copy() 浅拷贝调用方传入的列表，避免引擎内部 append 时污染外部数据
        self.token_ids = copy(token_ids)
        # 缓存最后一个 token：decode 阶段每步只需喂最后一个 token，避免每次 token_ids[-1]
        self.last_token = token_ids[-1]
        # 当前总 token 数（prompt + 已生成），decode 时随 append_token 递增
        self.num_tokens = len(self.token_ids)
        # prompt 长度，用于区分 prompt 部分和生成部分
        self.num_prompt_tokens = len(token_ids)
        # 已命中 prefix cache 的 token 数：这些 token 的 KV 已在缓存 block 里，无需重算
        self.num_cached_tokens = 0
        # 本轮调度实际要送进模型的 token 数：prefill 可能被 chunked 切分，decode 恒为 1
        self.num_scheduled_tokens = 0
        # 标记当前这一步是 prefill 还是 decode（被抢占后重新 prefill 时会重置为 True）
        self.is_prefill = True
        # block table：该序列的 KV 存在哪些 block 中的 block_id 列表，按逻辑顺序排列
        self.block_table = []
        # 从采样参数里摘出推理真正需要的三项，避免整包传给 worker
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    # 让 len(seq) 直接返回 token 数，调度器里写起来像操作一个 token 列表
    def __len__(self):
        return self.num_tokens

    # 让 seq[a:b] / seq[i] 直接切片底层 token_ids；
    # prepare_prefill 里用 seq[start:end] 取本轮要计算的 token 区间
    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        # 只读判断：是否已结束（引擎循环里用它决定何时把结果交给用户）
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        # 已生成的 token 数 = 总数 - prompt 数；用于判断是否达到 max_tokens
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        # 前半段是 prompt：切片返回 prompt 的 token id
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        # 后半段是生成结果：返回已生成的 token id
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        # 存下 num_tokens 个 token 需要多少个 block：向上取整除法，
        # (n + b - 1) // b 是不用 math.ceil 的标准整数写法
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        # 最后一个（可能未满）block 里已有多少 token：
        # 总数减去前面所有满 block 占掉的额度；decode 写 KV 时用它算 slot
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        # 返回第 i 个 block 覆盖的 token id 列表（不足一个 block 时就是剩余全部）。
        # prefix cache 用它作为 block 级哈希的输入
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        # decode 出一个新 token 后：追加进序列、更新缓存字段、总数 +1。
        # 注意此时 KV 还没写进 block（写 KV 发生在下一步 forward 的 store_kvcache 里）
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        # 自定义 pickle 协议：TP 多进程时序列对象要经共享内存发给各 rank，
        # decode 阶段只需要 last_token 一个 id，不必把整个 token_ids 列表传过去，
        # 大幅削减 IPC 数据量（多卡 prefill 时 token_ids 才是必需的，用于重算 KV）
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        # 与 __getstate__ 配对的反序列化：按同样的元组顺序还原字段
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            # prefill 情形：传来了完整 token_ids，直接恢复并重建 last_token
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # decode 情形：只传来了最后一个 token，token_ids 留空（worker 端 decode 用不到完整历史）
            self.token_ids = []
            self.last_token = last_state
