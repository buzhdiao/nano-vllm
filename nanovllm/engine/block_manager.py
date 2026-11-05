from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


# Block：KV cache 分页的最小管理单元。物理上每个 block 对应 KV cache 大张量里的
# [block_size, num_kv_heads, head_dim] 一段连续显存；这里只维护它的元数据
class Block:

    def __init__(self, block_id):
        # 物理块编号，即 KV cache 大张量第 0 维（block 维）的下标
        self.block_id = block_id
        # 引用计数：多少条序列正在共用这个 block（prefix cache 复用时 >1）
        self.ref_count = 0
        # 该 block 内容的哈希值（-1 表示尚未写入有效内容/不可复用）
        self.hash = -1
        # 该 block 覆盖的 token id 列表，与哈希配套用于“哈希相同再校验内容”防碰撞
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        # block 被写满并计算哈希后登记内容，从此可被其他序列按前缀复用
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        # 从空闲池取出复用前的清理：引用计数归 1（分配即被一条序列持有），
        # 哈希清 -1 表示旧内容作废，token_ids 清空
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    # 纯 CPU 侧的页表管理器（只管元数据，不碰显存张量），
    # 职责：分配/回收 block、引用计数、以及 block 级 prefix cache 的哈希复用

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # 所有物理块的元数据数组，block_id 即下标
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 哈希 -> block_id 的映射：prefix cache 的查找表，相同前缀哈希直接命中
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲块 id 池：双端队列，popleft 分配 / append 回收
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 已被占用的块 id 集合，O(1) 判断某块是否“在用”（与空闲池互补）
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 计算“前缀链式哈希”：hash(block_i) = H(prefix_hash || token_ids_i)。
        # 链式设计保证只有前面每个 block 都相同，后续 block 的哈希才可能相同，
        # 天然实现了“整条前缀逐块一致”的语义，而不必比较整条序列
        h = xxhash.xxh64()
        if prefix != -1:
            # 把父块哈希的 8 字节小端表示掺进本轮哈希，形成链
            h.update(prefix.to_bytes(8, "little"))
        # xxhash 是非加密哈希，速度远快于 hashlib，对齐内存读取极快；
        # numpy 转成连续字节流后一次性喂入
        h.update(np.array(token_ids).tobytes())
        # 返回 64 位整数摘要，作为 hash_to_block_id 的 key
        return h.intdigest()

    def _allocate_block(self) -> int:
        # 从空闲池队首取一个物理块
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        # 空闲池里的块引用计数必为 0，防御性断言
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            # 该块曾被哈希登记过（是潜在的 prefix cache 条目），
            # 现在要被覆写复用，必须先从查找表摘除，避免命中到内容已变的块
            del self.hash_to_block_id[block.hash]
        # 清理引用计数/哈希/内容，引用计数置 1（分给当前序列）
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        # 引用计数归零后真正回收：移出使用集、排回空闲池尾部
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        # prefill 前的预检：数一数 seq 的前缀能复用多少个缓存 block，以及空闲块是否够用
        h = -1
        num_cached_blocks = 0
        # 先假设所有 block 都要新分配
        num_new_blocks = seq.num_blocks
        # 注意只遍历前 num_blocks-1 个 block：最后一个块往往未写满，
        # 未满块的内容会继续增长，缓存“半块”需要额外屏蔽逻辑，这里直接不缓存
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            # 链式计算第 i 块哈希
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 哈希未命中，或命中但内容不同（哈希碰撞/被覆写）则停止：前缀复用只能连续
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                # 命中的块正被别的序列占用：共享即可，不需要新分配
                num_new_blocks -= 1
            # （若命中的块在空闲池里，它仍要被“取出”占用，所以不减 new_blocks）
        if len(self.free_block_ids) < num_new_blocks:
            # 空闲块不足以放下这条序列：返回 -1 让调度器停止继续收序列
            return -1
        # 返回可复用的前缀块数
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        # 为新序列建立完整 block table：前 num_cached_blocks 块复用缓存，其余新分配
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            # 重新沿链算哈希定位缓存块（can_allocate 已验证过命中有效）
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                # 块在用：引用计数 +1，实现多序列共享同一份物理 KV
                block.ref_count += 1
            else:
                # 块在空闲池里（早前某序列释放后留下的缓存内容）：
                # 手动“复活”它——从空闲池移除、加入使用集
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            # 缓存块直接挂进 block table；KV 数据本体已在显存里，无需拷贝
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            # 剩余前缀不命中：逐个分配全新块
            seq.block_table.append(self._allocate_block())
        # 记录命中前缀的 token 数：调度器据此只把“未缓存”的部分送进模型算
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        # 释放序列占用的全部块：逆序遍历只是习惯，关键是引用计数减到 0 才真正回收
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                # 无人引用才回空闲池；若 >0 说明还有别的序列在共享这块 KV
                self._deallocate_block(block_id)
        # 注意：块的 hash/token_ids 保留，块虽进空闲池但内容仍可被将来的 prefix cache 命中
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # decode 前检查：本步要不要写新 block、空闲块够不够。
        # len(seq) % block_size == 1 表示“刚跨进一个新块”（如 257 个 token 时第 257 个
        # 要落进新块的第一格），需要一个空闲块；不需要时表达式为 False(0)，恒成立
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # 与 can_append 的判断一致：只有当新 token 开启了新 block 时才真正分配物理块，
        # 其余情况新 KV 会写进当前块剩下的空位，不额外占块
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        # forward 写完 KV 后，把“本轮新写满”的块登记进 prefix cache 查找表
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        # 本轮没有跨过任何块边界（没有新写满的整块）则无事可做
        if start == end: return
        # 链头取上一块的哈希（start>0 时），从 -1 起链则表示从序列头开始
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            # 沿链续哈希：只有完整一致的前缀链才会产生相同哈希
            h = self.compute_hash(token_ids, h)
            # 登记内容 + 注册查找表，此后其他序列即可共享这块 KV
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
