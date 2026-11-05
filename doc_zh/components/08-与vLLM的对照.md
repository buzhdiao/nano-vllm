# 08 - 与 vLLM 的对照：这份代码省掉了什么

nano-vLLM 的定位是"用约 1300 行读得懂的 Python 复刻 vLLM 的推理内核"。它的调度器、PagedAttention、TP、prefix cache、CUDA Graph 都与 vLLM 同构，但作为一个**离线推理库**，它有意识地砍掉了服务化引擎的全部工程外壳。本篇列出"有什么、没什么"，方便读者带着 vLLM 的知识来读这份代码。

## 保留的内核机制（与 vLLM 对应）

| 机制 | nano-vLLM | vLLM 对应物 |
|---|---|---|
| 分页 KV cache | BlockManager（engine/block_manager.py） | vLLM block manager / paged KV |
| prefix cache | 块级 xxhash 链式哈希复用 | prefix caching（v1 默认开启） |
| 调度 | FIFO + prefill 优先 + recompute 抢占 + chunked prefill | v1 scheduler（更复杂的多级策略） |
| TP | Megatron 列/行并行 + 词表并行 | 同构，另有 custom all-reduce、EP、PP |
| CUDA Graph | 桶形 bs 捕获回放（decode 专用） | full cudagraph / piecewise |
| torch.compile | 局部装饰 RMSNorm/RoPE/激活/采样 | 全图 piecewise 编译 |
| 采样 | 温度 + 指数竞赛采样 | 全家族采样（top-p/top-k/beam…） |

## 省掉了什么

**1. 服务层与在线请求管理。** 没有 FastAPI/gRPC/OpenAI API server，没有 `AsyncLLMEngine`、请求队列、超时重试、流式增量输出。`LLM.generate()` 是同步阻塞的离线批量接口：所有请求一次性注入，循环跑到空（llm_engine.py:85-124）。请求到达的随机性、优先级、公平性统统不存在——调度器只面对一个 FIFO 双端队列。

**2. 连续批处理的在线语义。** vLLM v1 中每步可以从"新到达请求"插入 prefill；nano-vLLM 的"在线"只是 `add_request` 随时可调，`step()` 之间没有外部事件源。

**3. 高级采样与 stopping。** 只有 temperature + max_tokens + ignore_eos（sampling_params.py）。没有 top-p/top-k、 repetition penalty、logit bias、n-best、beam search、stop 字符串；`SamplingParams.__post_init__` 甚至直接禁止 temperature=0 的贪心解码（采样实现依赖除法，见 layers/sampler.py:17 的指数竞赛技巧）。

**4. 内存与量化的进阶机制。** 没有 KV cache 的 CPU swap（抢占一律 recompute 重算，scheduler.py:111-117）、没有 preemption mode 切换、没有 FP8/INT8/AWQ/GPTQ 量化路径、没有 sliding window/chunked attention 变体（RoPE 也只支持整头旋转，rotary_embedding.py:34）。

**5. 分布式的进阶机制。** TP 只支持 1-8 卡单机（config.py:39），固定 `tcp://localhost:2333` rendezvous（model_runner.py:31）；没有流水并行、专家并行、多节点、custom all-reduce（NCCL 默认实现）、没有 ray/k8s 编排。

**6. 模型生态。** 只内置 Qwen3（models/qwen3.py）。没有 HF Transformers 前向复用、多模型注册表、多模态、encoder-decoder、LoRA、speculative decoding、structured output（guided JSON）。

**7. 工程保障。** 没有 metrics/stats 日志、logits processor 钩子、KV cache 事件 tracing、兼容层（OpenAI server 的 /v1 接口）、大量 pytest 与 CI、版本化的 plugin 体系。

## 为什么砍了反而快

README 的 benchmark 显示 nano-vLLM 吞吐略胜 vLLM（1434 vs 1362 tok/s）。离线场景下"批填得满"比"调度得聪明"更重要：无服务层开销、每步 Python 路径极短、CUDA Graph 覆盖 decode、compile 融合小算子，四个因素把 CPU 端开销压到最低。这也是它作为教学代码的最大价值：**机制同构，复杂度降一个数量级**——读完这 1300 行，再看 vLLM 源码时就能把每一层包装识别出来。

## 建议的对照阅读路径

1. llm_engine.py:68-77 的 `step()` ↔ vLLM `LLMEngine.step()`
2. scheduler.py:35-108 ↔ vllm/core/scheduler.py 的 `schedule()`
3. block_manager.py 的哈希复用 ↔ vllm prefix caching（hash of block tokens）
4. model_runner.py:317-363 ↔ vllm worker 的 cudagraph 捕获
5. layers/linear.py ↔ vllm/model_executor/layers/linear.py（类名都一致）
