# 07 - Triton store_kvcache 内核解读

写 KV cache 是推理引擎里最不起眼却最频繁的操作：每一步、每一层都要把新算出的 K/V 精确塞进分页 cache 的对应格子。用 PyTorch 高级索引实现要经一堆 index kernel 和边界检查；nano-vLLM 用约 30 行 Triton（`layers/attention.py:13-40`）把它变成一个裸拷贝 kernel。

## kernel 源码

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                         k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
                         D: tl.constexpr):
    idx = tl.program_id(0)                    # 第 idx 个 program 处理第 idx 个 token
    slot = tl.load(slot_mapping_ptr + idx)    # 读目标 slot（cache 内一维下标）
    if slot == -1: return                     # CUDA Graph 桶内补位行：跳过
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)      # 从新 K 里读一行
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)   # 写进 cache 第 slot 行
    tl.store(v_cache_ptr + cache_offsets, value)
```

要点逐条拆解：

**1. 并行模型**：grid 是一维 `(N,)`，N = 本批 token 数。每个 program（可理解为一个 CUDA block）负责一个 token 的整行 K/V 拷贝，`tl.program_id(0)` 取自己的编号（attention.py:24）。

**2. slot 即一维页内地址**：`slot_mapping` 由 ModelRunner 按 `block_table[t // block_size] * block_size + t % block_size` 算好（model_runner.py:208-228），kernel 拿到的已经是"cache 大矩阵的行号"，`slot * D + arange(0, D)` 就是元素偏移（attention.py:37）。**分页的全部复杂性都留在 CPU 侧打包阶段**，GPU 侧只做散射写。

**3. `D: tl.constexpr`**：行宽 = `num_heads * head_dim` 在编译期常量展开，`tl.arange(0, D)` 生成行内偏移向量，整行 load/store 向量化完成（attention.py:28-31）。TP 下每 rank 的 D 随 KV 头数减半而变小。

**4. stride 手工寻址**：新 K/V 是 `[N, heads, dim]` 三维张量，kernel 把它当 `[N, D]` 二维视图：第 idx 行起点是 `idx * key_stride`，其中 `key_stride = key.stride(0)`。Python 侧的四个 assert（attention.py:50-55）保证"最后一维连续、头间无 padding、cache 行宽恰为 D"这些布局假设成立——断言失败即布局不符，宁可崩也不写坏 cache。

**5. `slot == -1` 早退**：CUDA Graph 回放时桶尺寸向上取整，多余行的 slot 必须填 -1（model_runner.py:293-294），kernel 见 -1 直接 return，保证"图是死形状、批是活大小"不写脏 cache（attention.py:27）。

## 启动侧封装

`store_kvcache`（attention.py:43-58）只做形状/stride 校验，然后 `store_kvcache_kernel[(N,)](...)` 以方括号语法传 grid 启动。k_cache/v_cache 虽然逻辑上是 `[num_blocks, block_size, heads, dim]` 四维，但 Triton 只收指针，四维连续内存天然可以按 `[num_blocks*block_size, D]` 扁平寻址——前提正是 assert 验证过的连续布局。

## 性能定位

这是一个**纯带宽型** kernel：读 N×D×2 个元素、写 N×D×2 个元素，无计算。相比 PyTorch 的 `index_copy_` / 高级索引方案，省去了中间索引张量的构造与多次 kernel launch；对 decode（每步 N = bs，每层都要调一次）尤其友好，也因为它没有动态形状，能被 CUDA Graph 捕获后随图重放。
