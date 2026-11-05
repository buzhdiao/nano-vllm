# 04 - ModelRunner 与 CUDA Graph 捕获回放

ModelRunner（`nanovllm/engine/model_runner.py`，363 行）是每个 GPU 一个的执行器：rank 0 在主进程，其余在 spawn 子进程常驻循环里（见 05 篇）。本篇聚焦它的四件核心工作：显存预算、输入打包、CUDA Graph、执行流水。

## 初始化顺序（model_runner.py:20-64）

`init_process_group("nccl")` → `set_device(rank)` → 切默认 dtype/device 建模并加载权重 → **warmup** → `allocate_kv_cache` → 捕获 CUDA Graph。顺序不能乱：warmup 的显存峰值是 KV cache 预算的输入，KV cache 分配又是图捕获的前提。

**warmup**（model_runner.py:123-138）构造"最大批 × 最长序列"的假 prefill 跑一遍，让 cuBLAS/cuDNN 工作区、kernel 编译缓存全部就位，并测出分配器的峰值占用。

## KV cache 显存预算（model_runner.py:140-161）

```python
config.num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
```

- `total/used` 来自 `torch.cuda.mem_get_info()`（驱动视角，含其他进程）；
- `peak - current` 是 PyTorch 分配器"曾用过的峰值减去现在持有的"——warmup 的临时激活已被释放但缓存池还占着，扣除后按最坏情况估计；
- `block_bytes = 2 × 层数 × block_size × kv头数 × head_dim × dtype字宽`，TP 下每 rank 只分到 `num_kv_heads // world_size` 个头。

随后一次性 `torch.empty` 预分配 `[2, 层数, 块数, block_size, 头数, 头维]` 的大张量，按层切片挂到每个 `Attention` 模块（model_runner.py:161-170）。运行期零显存申请，这是 PagedAttention 稳定性（不会 OOM）的前提。

## 输入打包

- **prefill**（model_runner.py:180-241）：把批内所有 token 扁平拼接成 `[Σtokens]`，`cu_seqlens_q/k` 记前缀和供 flash_attn varlen 切分；`positions` 是各 token 的绝对下标；`slot_mapping` 逐块算出每个新 token 的 cache 写入 slot（model_runner.py:208-228）。命中 prefix cache 时 `cu_seqlens_k[-1] > cu_seqlens_q[-1]`（model_runner.py:229），需要额外准备 block table。
- **decode**（model_runner.py:243-262）：每序列只喂 `last_token` 一个 id，`slot_mapping` 指向队尾空位，`context_lens` 与 block table 齐备。

所有张量都用 `pin_memory=True` 创建再 `.cuda(non_blocking=True)`——锁页内存让 H2D 拷贝走 DMA 且异步，CPU 侧打包下一批与传输重叠。

## CUDA Graph：为什么与怎么做

decode 每步只算 1 token/序列，批内数百个 kernel，launch 开销占比极高。CUDA Graph 把整段内核序列录制成一条可重放的图，回放时一次提交，launch 开销近乎归零。

捕获（model_runner.py:317-364）要点：

1. **桶形批尺寸** `[1,2,4,8,16,32,...,512]`（model_runner.py:333）：图形状固化，按桶向上取整覆盖所有 bs；
2. **从大到小捕获**并共享 `graph_pool` 内存池（model_runner.py:338-352），避免每张图独立保留中间激活；
3. 每张图必须先在普通流上 **warmup** 再 `with torch.cuda.graph(...)` 捕获（model_runner.py:344-347）；
4. 输入/输出全部预分配成固定地址缓冲区（model_runner.py:325-330）。

回放（model_runner.py:276-303）：找到 `>= bs` 的最小桶，把真实输入**原地拷进**固定缓冲区，多余行用 `slot_mapping=-1`（kernel 跳过写 cache）、`context_lens=0` 屏蔽，然后 `graph.replay()`（model_runner.py:301），从固定输出缓冲区取前 bs 行算 logits。

`enforce_eager=True` 或 bs>512 时退回普通 eager 前向（model_runner.py:278）。prefill 永远不走图——变长形状没法固化。

## 全局 Context

打包出的元数据经 `set_context` 放进模块级单例（utils/context.py），attention/lm_head 层在 forward 里自取，避免十几个参数层层透传。
