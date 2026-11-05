# 03 - BlockManager 与 PagedAttention 分页 KV cache 原理

传统推理为每条序列预留 `max_model_len` 长度的连续 KV 显存，内部碎片率可达 60-80%。**PagedAttention** 借鉴操作系统虚拟内存：把 KV cache 切成固定大小的 **block**（本仓库默认 256 token/块，config.py:29），按需分配，用 **block table**（页表）把逻辑位置映射到物理位置。BlockManager（`nanovllm/engine/block_manager.py`，179 行）是这套页表的纯 CPU 元数据管理器——它从不触碰显存张量本身。

## 三个数据结构

- `blocks: list[Block]`：所有物理块的元数据数组，`block_id` 即下标（block_manager.py:42）；
- `free_block_ids: deque`：空闲块池，`popleft` 分配 / `append` 回收（block_manager.py:46）；
- `hash_to_block_id: dict`：prefix cache 查找表（block_manager.py:44）。

每个 `Block` 记录 `ref_count`（引用计数，多序列共享缓存块时 >1）、`hash`、`token_ids`（block_manager.py:11-32）。

## block table 与 slot mapping

序列的第 i 个逻辑块存在 `seq.block_table[i]` 号物理块里（sequence.py:37、90-94）。于是"第 t 个 token 的 KV 写到哪"就是两级寻址：

```
slot = block_table[t // block_size] * block_size + t % block_size
```

`ModelRunner.prepare_prefill` / `prepare_decode` 按这个公式生成 `slot_mapping`（model_runner.py:208-228、243-262），Triton kernel 据此把新 K/V 散射进任意物理块（见 07 篇）。`can_append`/`may_append`（block_manager.py:152-162）用 `len(seq) % block_size == 1` 判断"新 token 开启了新块"，只有跨块时才真正分配——块内剩余空位天然免分配。

## prefix cache：块级哈希复用

核心是**链式哈希**（block_manager.py:51-63）：

```python
h = xxhash.xxh64()
if prefix != -1:
    h.update(prefix.to_bytes(8, "little"))   # 把父块哈希掺进来
h.update(np.array(token_ids).tobytes())
```

`hash(block_i) = H(hash(block_{i-1}) || token_ids_i)`。链式结构保证**前缀哈希相同 ⟺ 逐块内容都相同**，且任何一块不同，后续所有块哈希全变——天然实现"最长公共前缀"语义，无须比较整条序列。

- `can_allocate`（block_manager.py:86-111）沿链探测命中数；只扫前 `num_blocks-1` 块，**未写满的尾块不缓存**；命中块若在用则不必新分配。
- `allocate`（block_manager.py:113-137）把命中块直接挂进 block table 并增加引用计数（KV 数据本体已在显存，零拷贝共享），其余块走 `_allocate_block`；命中 token 数记进 `seq.num_cached_tokens`，调度器据此只算未缓存部分。
- `hash_blocks`（block_manager.py:164-179）在前向写完 KV 后，把"本轮新写满"的块沿链哈希并登记进查找表。

## 引用计数与回收

`deallocate`（block_manager.py:140-151）只把 `ref_count` 减到 0 才真正回空闲池；即便回了池，块的 `hash/token_ids` 仍保留，随时可被下一个请求"复活"命中——这就是为什么空闲块也要先在 `_allocate_block` 里把过期哈希从查找表摘除（block_manager.py:72-74）。共享块被抢占方释放时，持有方的 KV 不受影响。

flash_attn 侧按 block table 取数的部分见 06 篇。
