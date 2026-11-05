# 01 - LLMEngine 与生成循环

LLMEngine 是整个引擎的组装车间与总控循环（`nanovllm/engine/llm_engine.py`，共 125 行）。它做四件事：装配配置、拉起张量并行子进程、驱动"调度一步 → 前向一步"的循环、把结果还给用户。

## 构造流程

`__init__`（第 19 行起）先用 `dataclasses.fields` 内省 `Config` 的字段，把用户 kwargs 里属于配置的项过滤出来构造 `Config`（llm_engine.py:22-24）。随后若 `tensor_parallel_size > 1`，为 1..N-1 每个 rank 用 `mp.get_context("spawn")` 创建子进程，入口直接是 `ModelRunner` 构造函数，并给每个子进程配一个 `Event` 用于唤醒（llm_engine.py:32-39）。rank 0 的 `ModelRunner` 留在主进程（llm_engine.py:42）。最后加载 tokenizer、回填 `config.eos`（llm_engine.py:44-46）、注册 `atexit` 钩子保证子进程优雅退出（llm_engine.py:49）。

用 spawn 而非 fork 是因为 CUDA 上下文不能跨 fork 继承；`Event` 是共享内存信箱的"门铃"，配对逻辑见第 04 篇。

## step：引擎的心跳

`step()`（llm_engine.py:68-77）是引擎的最小执行单元：

```python
seqs, is_prefill = self.scheduler.schedule()
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

三行分别对应"决定算什么 → 真的去算 → 登记账本"。返回值里 `num_tokens` 用正负号编码批次类型：prefill 批返回本批 token 数（正），decode 批返回 `-len(seqs)`（负），让 `generate` 能分别统计两条吞吐曲线（llm_engine.py:73）。只有在本步刚好完成的序列才会进 `outputs`（llm_engine.py:78）。

## generate：离线批量推理

`generate()`（llm_engine.py:85-124）是 vLLM 风格的离线入口：所有请求一次性 `add_request` 注入等待队列，然后 `while not self.is_finished(): step()` 空转到全部完成。与在线服务不同，它不模拟请求到达，只求把批填满、吞吐拉满——这也是 nano-vLLM 砍掉 HTTP 服务和连续批处理请求管理的前提。请求支持字符串或 token id 列表两种形态（llm_engine.py:60-65）；结果按 `seq_id` 排序，保证与输入顺序一一对应（llm_engine.py:122-124）。

进度条用 tqdm 的 `set_postfix` 同时展示 Prefill/Decode 两个吞吐（llm_engine.py:92、112-115），`bench.py` 里 `use_tqdm=False` 关闭以免拖慢计时。

## 阅读指引

- 批怎么组出来：见 02-调度器；
- `model_runner.call` 背后的多进程机制：见 04 与 05；
- 顶层门面 `LLM = LLMEngine` 只是别名（nanovllm/llm.py:6-8），对齐 `from vllm import LLM` 的习惯。
