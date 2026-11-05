import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


# 模型执行器：每个 TP rank 一个实例（rank 0 在主进程，其余在子进程）。
# 职责：构建/加载模型、分配 KV cache 显存、把调度结果打包成张量、
# 驱动 prefill/decode 前向与采样、管理 CUDA Graph 的捕获与回放
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        # rank 0 持有“所有子进程的 Event 列表”（用于广播），子进程只持自己的那个
        self.event = event

        # 初始化 NCCL 进程组：TP 各 rank 通过 TCP rendezvous 互相找到（本机端口 2333）
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # 把当前进程绑定到自己的显卡：TP 每个进程管一张卡
        torch.cuda.set_device(rank)
        # 记住全局默认 dtype，稍后恢复
        default_dtype = torch.get_default_dtype()
        # 按模型配置切默认 dtype（如 bf16）：之后新建的层/参数/张量都用它
        torch.set_default_dtype(hf_config.dtype)
        # 默认设备切到 cuda：层参数直接在 GPU 上创建，免去逐个 .cuda()
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        # 从 safetensors 加载权重（含 TP 切分逻辑）
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # 预热：先跑一次最大批的 prefill，让 cudnn/cublas 句柄、编译缓存、显存碎片都就位，
        # 也为下一步估算“模型本身占了多少显存”提供峰值数据
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            # 非 eager 模式：为 decode 捕获一组 CUDA Graph
            self.capture_cudagraph()
        # 恢复全局默认 device/dtype，避免污染之后创建的 tokenizer 等对象
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # 主 rank 创建 1MB 的 POSIX 共享内存段，作为向子进程发号施令的“信箱”
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                # 子进程先等 rank 0 创建完成（barrier 同步），再以附属模式打开同名段
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                # 子进程进入常驻循环：等主 rank 的指令并执行，直到收到 "exit"
                self.loop()

    def exit(self):
        if self.world_size > 1:
            # 先关闭本进程的共享内存映射，barrier 保证大家都在 unlink 之前关完
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                # 只有创建者负责 unlink，真正释放共享内存段
                self.shm.unlink()
        if not self.enforce_eager:
            # 释放 CUDA Graph 及其内存池
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        # 销毁 NCCL 进程组，各进程才能干净退出
        dist.destroy_process_group()

    def loop(self):
        # 子进程主循环：阻塞等事件 -> 读共享内存里的 (方法名, 参数) -> 执行 -> 再等
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                # 收到退出指令：结束循环，随后 __init__ 返回、进程自然退出
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        # 阻塞直到主 rank set() 事件（信箱里有新信）
        self.event.wait()
        # 前 4 字节是小端存储的 payload 长度 n
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # payload 是 pickle 序列化的 [方法名, *参数]；buf 是 memoryview，切片即视图
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        # 清除事件：下一次 wait() 将再次阻塞，等下一封信
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        # 序列化调用（含 Sequence 列表——Sequence 自定义了 __getstate__ 精简传输）
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 写长度头，再写 payload；注意先写长度后写内容，读者以长度为准截取
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        # 广播：set 所有子进程的事件，唤醒它们各自读信
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            # TP>1 时主 rank 先把调用广播出去，让所有 rank 执行同一个方法
            self.write_shm(method_name, *args)
        # 反射取方法并调用：rank 0 正常执行并拿到返回值，子进程执行同样的方法
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        # 清掉加载权重后残留的缓存，并重置峰值统计，让下面的测量只反映真实前向开销
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 构造“最大规模”的假批次：序列长取预算与最大模型长的较小者
        seq_len = min(max_num_batched_tokens, max_model_len)
        # 能塞下几条这样的序列（至少 1 条），并夹在 max_num_seqs 内
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            # 直接标记整条序列为待算 token，假装一次超大 prefill
            seq.num_scheduled_tokens = seq_len
        # 用假数据跑一次真实前向：激活 cudnn 工作区、触发 kernel 编译缓存
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        # 向驱动查询：总显存 / 当前已用
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # 本进程 torch 分配器的峰值与当前持有量（峰值 - 当前 = 可归还但暂占着的部分）
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # TP 切分后每个 rank 分到的 KV 头数（num_kv_heads 必须能被 world_size 整除）
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 兼容没有显式 head_dim 的模型：用 hidden_size / num_heads 推出
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 单个 block 的字节数：2(K 和 V) * 层数 * 每块 token 数 * 头数 * 头维 * dtype 字宽
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        # 可用块数 = (目标显存占用 - 系统已用 - 模型峰值 + 分配器当前持有) // 单块字节。
        # 减 peak 加 current 的含义：峰值里有一次性激活/临时量，扣除后才是稳定可用的部分
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 一次性预分配整块 KV cache：形状 [2(K/V), 层数, 块数, 块大小, KV 头数, 头维]。
        # 预分配后运行期零显存申请，这是 PagedAttention 稳定性的前提
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # 把大张量按层切片（视图，不拷贝）挂到每个 Attention 层上：
                # k_cache = kv_cache[0, layer_id]，即该层 K 的 [块数, 块大小, 头数, 头维]
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        # 批内 block table 长度不齐：补 -1 到同一长度（flash_attn 用 -1 标记无效页）
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # CPU 端建张量并 pin memory：锁页内存可用 DMA 异步拷贝，
        # .cuda(non_blocking=True) 让 H2D 传输与后续 CPU 工作重叠
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        # 把一个 prefill 批打包成 flash_attn varlen 需要的“扁平拼接 + 累积长度”格式
        input_ids = []
        positions = []
        # cu_seqlens：累积序列长度（前缀和），flash_attn 以此在扁平张量里切出各序列
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        # slot_mapping：本批每个新 token 的 KV 要写进 cache 的绝对位置（block_id*block_size+块内偏移）
        slot_mapping = []
        # prefill 无需整表寻址，但命中 prefix cache 时例外（见函数尾部）
        block_tables = None
        for seq in seqs:
            # 从“已缓存 token 数”开始算：prefix cache 命中的部分跳过不重算
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            # K 的可见长度 = 已缓存 + 本步新算：causal 下第 i 个 token 能看到 [0, i] 的全部 KV
            seqlen_k = end
            # 取本步的 token 区间，扁平追加进批
            input_ids.extend(seq[start:end])
            # 位置编码：每个 token 在序列中的绝对下标（RoPE 按它查 cos/sin 表）
            positions.extend(range(start, end))
            # 累积长度追加（前缀和）
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                # warmup 假序列没有真实 block table：只算形状，不写 cache
                continue
            # 逐块计算本步 token 的 cache slot 区间：
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                # 该块在 cache 里的起始 slot = 物理 block_id * 每块 token 数
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    # 起始块：从块内偏移 start%block_size 开始（前面 token 已在缓存里）
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    # 中间块：整块写满
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    # 末块：写到本步终点在本块内的偏移
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                # 连续的 slot 区间展开成单个下标列表
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            # K 比 Q 长：说明命中了 prefix cache，Q 的 KV 只在 cache 里，
            # 注意力要按 block table 去 cache 里取 K/V
            block_tables = self.prepare_block_tables(seqs)
        # 全部元数据一次性建张量上卡：pin_memory + non_blocking 异步传输
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 写入全局上下文：attention 层的 forward 从这里取元数据（避免层层传参）
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        # decode 批：每条序列只有 1 个新 token，打包成 [batch] 形状
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # 只喂最后一个 token（历史 KV 已在 cache 里）
            input_ids.append(seq.last_token)
            # 新 token 的位置 = 当前长度 - 1
            positions.append(len(seq) - 1)
            # 注意力可见的历史长度（含本 token）
            context_lens.append(len(seq))
            # 本 token 的 KV 写入位置：最后一个块的块内偏移。
            # +len 后再 -1：即“追加前长度”指向的空位
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        # 同 prefill：锁页内存 + 异步上卡
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # decode 必须按 block table 从 cache 读全部历史 K/V
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        # 采样只需每条序列的温度；只有 rank 0 需要采样
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # inference_mode：禁用版本计数与 autograd 追踪，比 no_grad 更省开销
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # prefill / eager 模式 / 超大批（超出 graph 覆盖范围）：直接 eager 前向 + 取 logits
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # decode：走 CUDA Graph 回放。内核参数已固化，只更新输入缓冲区
            bs = input_ids.size(0)
            context = get_context()
            # 从预捕获的桶里找 >= bs 的最小图（向上取桶，多余行填 0 参与计算但结果被丢弃）
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 把真实输入拷进捕获时的固定地址缓冲区（这是图回放的唯一“输入接口”）
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # slot_mapping 先整体填 -1（store_kvcache 对 -1 跳过写 cache），再覆盖前 bs 行：
            # 桶比实际批大的部分绝不能写 cache
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            # context_lens 补 0：多出的行注意力长度为 0，不产生有效输出
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            # block_tables 只覆盖有真实数据的列（各序列表长可能不同）
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 一条 GPU 命令重放整段捕获好的内核序列
            graph.replay()
            # 输出缓冲区同样固定地址：取前 bs 行算 logits
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 一步执行的完整流水：打包输入 -> (rank0) 准备采样参数 -> 前向 -> 采样 -> 清理上下文
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        # 只有 rank 0 采样并转成 Python list；其他 rank 返回 None（调度只在主进程）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 上下文只在单次 forward 内有效：用完立刻复位，防止下一次调用读到脏元数据
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        # 图覆盖的最大 batch：不超过 max_num_seqs，也不超过 512
        max_bs = min(self.config.max_num_seqs, 512)
        # 最大序列长折算的最大块数：block_tables 缓冲区的列数
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 预分配所有“图输入/输出缓冲区”：捕获后地址固化，回放时只改内容不改地址
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 桶形 batch 尺寸：小 bs 密集（1/2/4/8），大 bs 按 16 递增，
        # 兼顾“向上取桶”的Padding 浪费与图数量/显存开销
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        # 共享内存池：所有图共用一个 allocator 池，避免每张图独立保留中间激活显存
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            # 从大到小捕获：先给最大图分配显存，小图能更好地复用池中块
            graph = torch.cuda.CUDAGraph()
            # 捕获前把上下文设成 decode 形态（graph 输入即这些缓冲区的切片）
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # 必须先 warmup：让 kernel 先在普通流上编译/分配好，捕获流上才不会有意外分配
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                # 捕获阶段模型并不真正执行：只是把内核调用连同固定指针录进图
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                # 第一个（最大的）图的内存池作为共享池
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存全部缓冲区引用：回放前由 run_model 原地更新它们
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
