from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


# 调度器：每一步决定“哪些序列、算哪些 token”组成一个批次送给 ModelRunner。
# 核心策略：prefill 优先（新请求尽快出第一个 token），decode 靠后；显存不足时抢占
class Scheduler:

    def __init__(self, config: Config):
        # 批内最大序列数（vLLM 同名参数）：decode 批的上限
        self.max_num_seqs = config.max_num_seqs
        # 单批最大 token 预算：prefill 批的显存/算力上限，也是 chunked prefill 的切分依据
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # EOS id，postprocess 里判断自然结束
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        # 块管理器：负责 KV cache 的物理块分配与 prefix cache
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # waiting：已提交但还没 block 的序列（deque 支持队首插入，抢占时用）
        self.waiting: deque[Sequence] = deque()
        # running：已获得 block、正在 decode 的序列
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        # 两个队列都空了，引擎的生成循环才结束
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # 新请求进入等待队列尾部（FIFO admission）
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        # 本批已用掉的 token 预算
        num_batched_tokens = 0

        # prefill：优先填新请求。返回值第二位 True 表示本批是 prefill 批
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # 只看队首但不弹出：放不下（chunked）时它要留在队首
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                # token 预算用尽，本批收工
                break
            if not seq.block_table:
                # 序列还没分配过 block（首算）：先做 prefix cache 预检
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # 空闲块不够容纳这条序列：为避免死锁，本批不再继续收 prefill，
                    # 把预算留给 decode（也能间接催促 running 序列释放显存）
                    break
                # 需要真正计算的 token = 总长 - 命中缓存的长度（缓存部分 KV 已在显存）
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # seq 已有 block_table：是被抢占后重新调度的序列，
                # 上次算到哪（num_cached_tokens）之后的部分需要重算（recompute 恢复策略）
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                # 这条序列放不进剩余预算：仅当它已是本批第一条时允许 chunked 切分，
                # 否则直接 break——保证批内至多一条被切分的序列，cu_seqlens 逻辑简单
                break
            if not seq.block_table:
                # 通过预检：按缓存命中情况正式分配物理块、建立 block table
                self.block_manager.allocate(seq, num_cached_blocks)
            # 本步实际要算的 token 数：min(需要算的, 剩余预算)——chunked prefill 的切分点
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # 本轮能把它整个算完：状态转 RUNNING、从 waiting 挪进 running
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            # 未算完的 chunked 序列留在 waiting 队首，下一轮继续从断点算
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            # 只要收到了任何 prefill，就跳过 decode：prefill 优先，新请求尽快产出首 token
            return scheduled_seqs, True

        # decode：对 running 队列里的序列各算一个 token
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            # 队首弹出；若最终未被抢占则还会放回去
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # 空闲块不足以再写一个新块：必须抢占别的序列腾显存。
                # 牺牲者选 running 队尾（pop：最晚调度、完成进度最接近的按 LIFO……
                # 实际效果是保留队首已等最久的序列）
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    # 队列里只剩自己还不够：抢占自己（整体回到 waiting 重算）
                    self.preempt(seq)
                    break
            else:
                # while 未 break，即显存充足：本步只调度 1 个 token（decode 的定义）
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                # 可能为其分配新块（跨块边界时）
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        # 兜底断言：走到 decode 分支至少要调度出一条序列，否则引擎逻辑有 bug
        assert scheduled_seqs
        # 把本批序列按原顺序放回 running 队首：extendleft 会逆序插入，
        # 所以先 reversed 保证队序不变（被抢占牺牲的序列本来就从队尾消失）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # 抢占 = 撤销其全部物理块、状态打回 WAITING、插到 waiting 队首（下次最先重算）。
        # nano-vllm 采用 vLLM 的 recompute 策略：不换出 KV 到 CPU，直接丢弃，
        # 重新调度时按“未算部分”重算 prefill——实现简单且对吞吐影响小
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        # forward + 采样完成后：登记缓存、推进进度、追加新 token、判定结束
        for seq, token_id in zip(seqs, token_ids):
            # 把本轮新写满的块注册进 prefix cache 查找表（供后续请求复用）
            self.block_manager.hash_blocks(seq)
            # 已算完的 token 计入“已缓存”：调度器据此知道下次从哪继续
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # chunked prefill 尚未算完：本轮没有采样出“下一个 token”，直接进入下一轮
                continue
            # prefill 算完（或 decode 一步）：把采样结果追加为新 token
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # 命中 EOS（且未被 ignore_eos 关闭）或达到 max_tokens：序列完成，
                # 立即释放全部物理块归还空闲池，供后续 prefill 使用
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
