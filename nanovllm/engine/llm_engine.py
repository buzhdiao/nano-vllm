import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


# 引擎顶层：组装配置、TP 子进程、分词器、调度器与 0 号卡执行器，
# 并向用户提供 generate() 这个“离线批量推理”入口
class LLMEngine:

    def __init__(self, model, **kwargs):
        # 用 dataclasses.fields 内省 Config 的字段名集合，
        # 把用户 kwargs 里与配置相关的项过滤出来（其余参数直接忽略，容错）
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # block_size 同时是序列切块与 block table 的基准，挂到 Sequence 类属性上全局共享
        Sequence.block_size = config.kvcache_block_size
        # ps：TP 1..N-1 号 rank 的子进程列表；events：主 rank 通知各子进程“有新任务”的事件
        self.ps = []
        self.events = []
        # 显式用 spawn 上下文：CUDA 不允许 fork 出的子进程继承已初始化的上下文，
        # spawn 重新 import 模块、干净地创建进程，是多卡 PyTorch 的标准做法
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            # 每个子进程一个 Event：主进程 set() 广播“共享内存里已有新请求”，子进程 wait()
            event = ctx.Event()
            # 子进程入口直接是 ModelRunner 构造函数，构造完会进入 loop() 常驻等待
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # rank 0 的执行器在主进程内跑：调度器只存在于主进程，输出 token 也由 rank 0 采样
        self.model_runner = ModelRunner(config, 0, self.events)
        # fast tokenizer：多请求批量编码更快
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # EOS 由 tokenizer 决定（与模型 config 的 eos 可能不同），回填进 config 供调度器判停
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # 进程退出时确保 TP 子进程被优雅关闭（否则 spawn 的子进程可能残留占用显卡）
        atexit.register(self.exit)

    def exit(self):
        # 先通知各 rank 退出（rank 0 会把 "exit" 调用经共享内存广播给所有子进程）
        self.model_runner.call("exit")
        # 删除引用触发 rank 0 资源清理
        del self.model_runner
        for p in self.ps:
            # join 等待子进程真正结束
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 支持两种入参：字符串（先分词）或已编码的 token id 列表（bench 用它绕过分词开销）
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        # 进入调度器 waiting 队列
        self.scheduler.add(seq)

    def step(self):
        # 引擎的最小执行单元：调度一批 -> 前向 + 采样 -> 后处理（登记缓存/追加 token/判停）
        seqs, is_prefill = self.scheduler.schedule()
        # 吞吐统计约定：prefill 批返回“本批计算的 token 数”（正数），
        # decode 批返回“-批内序列数”（负数），generate() 据此区分两条吞吐曲线
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # rank 0 执行；TP>1 时经由共享内存把同样的调用广播给其他 rank
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 只收集本步刚好完成的序列：(seq_id, 已生成 token ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        # 所有序列都已结束（队列为空）则整体生成完成
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # 进度条总数 = 请求数；disable 控制静默（bench 场景关掉以免拖慢计时）
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            # 单个采样参数广播成与 prompts 等长的列表，统一后续 zip 的处理路径。
            # 注意这是浅拷贝引用（同一对象重复引用），但采样参数只读，安全
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            # 全部请求一次性注入调度器（离线批处理：不模拟在线到达，只求吞吐）
            self.add_request(prompt, sp)
        # seq_id -> 生成 token ids；最后按 seq_id 排序还原与 prompts 一致的顺序
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                # prefill 吞吐：本批 token 数 / 本步耗时
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                # decode 吞吐：负号还原为“本步产出的 token 数”
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                # 每完成一条序列推进一格
                pbar.update(1)
        pbar.close()
        # 按 seq_id 升序取回结果，保证输出顺序与输入 prompts 一一对应
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 解码成文本，同时保留 token ids 供需要 id 的用户使用
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
