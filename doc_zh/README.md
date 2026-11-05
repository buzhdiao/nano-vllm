# doc_zh —— nano-vLLM 中文文档工程

本目录是 nano-vLLM 的中文配套文档，包含三部分：

1. **核心源码逐行中文注释**：`nanovllm/` 下全部 19 个 `.py` 文件均已加入逐行中文注释（只增注释、不改任何代码语义，经 token 级对比校验与原代码完全一致）。每个逻辑行上方或行尾说明"这行在做什么、为什么"，关键算法（block table 管理、slot_mapping、CUDA Graph 捕获回放、TP 切分、prefix cache 链式哈希等）均给出原理性解释。
2. **组件/原理文档 8 篇**：`components/`。
3. **Python 高级技法 100 讲**：`techniques-100.md`。
4. **README 中文翻译**：`translations/README.zh.md`。

## 阅读顺序建议

### 第一轮：跑通宏观流程（先建立整体图景）
1. [components/01-LLMEngine与生成循环.md](components/01-LLMEngine与生成循环.md) —— 引擎组装与 `step()` 心跳，先读 `nanovllm/engine/llm_engine.py`
2. [components/02-调度器与批处理.md](components/02-调度器与批处理.md) —— prefill/decode 两阶段、chunked prefill、抢占，配 `scheduler.py`
3. [components/03-BlockManager与PagedAttention分页KVCache.md](components/03-BlockManager与PagedAttention分页KVCache.md) —— block table、slot mapping、prefix cache 哈希复用，配 `block_manager.py`

### 第二轮：深入执行层（GPU 上到底发生了什么）
4. [components/04-ModelRunner与CUDAGraph捕获回放.md](components/04-ModelRunner与CUDAGraph捕获回放.md) —— 显存预算、输入打包、图捕获回放，配 `model_runner.py`
5. [components/06-注意力层与flash_attn集成.md](components/06-注意力层与flash_attn集成.md) —— varlen/with_kvcache 两个入口，配 `layers/attention.py`
6. [components/07-Triton_store_kvcache内核解读.md](components/07-Triton_store_kvcache内核解读.md) —— 逐行拆解唯一的手写 CUDA kernel

### 第三轮：横向能力（多卡与服务化对照）
7. [components/05-张量并行.md](components/05-张量并行.md) —— 列/行并行、通信原语、weight_loader、共享内存控制流，配 `layers/linear.py`、`layers/embed_head.py`、`model_runner.py` 的 shm 部分
8. [components/08-与vLLM的对照.md](components/08-与vLLM的对照.md) —— 系统性盘点"有什么、没什么"，适合最后收尾

### 随时穿插
- [techniques-100.md](techniques-100.md)：100 个 Python 高级技法/库讲解，每条都指向仓库真实代码。可当"读源码时的字典"用——读注释遇到不认识的写法（`for-else`、`extendleft(reversed(...))`、`weight_loader` 函数属性……）就翻对应条目。
- [translations/README.zh.md](translations/README.zh.md)：原 README 的中文翻译。

## 本次新增/修改文件总览

```
doc_zh/
├── README.md                                    # 本索引
├── techniques-100.md                            # Python 高级技法与重要库 100 讲
├── components/
│   ├── 01-LLMEngine与生成循环.md
│   ├── 02-调度器与批处理.md
│   ├── 03-BlockManager与PagedAttention分页KVCache.md
│   ├── 04-ModelRunner与CUDAGraph捕获回放.md
│   ├── 05-张量并行.md
│   ├── 06-注意力层与flash_attn集成.md
│   ├── 07-Triton_store_kvcache内核解读.md
│   └── 08-与vLLM的对照.md
└── translations/
    └── README.zh.md

nanovllm/            # 全部 19 个 .py 文件新增逐行中文注释（代码语义零改动）
├── __init__.py / llm.py / config.py / sampling_params.py
├── engine/  sequence.py · block_manager.py · scheduler.py · llm_engine.py · model_runner.py
├── layers/  activation.py · attention.py · embed_head.py · layernorm.py · linear.py · rotary_embedding.py · sampler.py
├── models/  qwen3.py
└── utils/   context.py · loader.py
```

> 提示：文档中的 `file:line` 行号均对齐"注释版"源码。
