<p align="center">
<img width="300" src="../../assets/logo.png">
</p>

# Nano-vLLM

一个从零开始构建的轻量级 vLLM 实现。

## 主要特性

* 🚀 **快速的离线推理** —— 推理速度可与 vLLM 相媲美
* 📖 **易读的代码库** —— 约 1,200 行 Python 代码的简洁实现
* ⚡ **优化套件** —— 前缀缓存（Prefix Caching）、张量并行（Tensor Parallelism）、Torch 编译、CUDA Graph 等

## 安装

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## 模型下载

如需手动下载模型权重，请使用以下命令：
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## 快速开始

用法参见 `example.py`。API 与 vLLM 的接口保持一致，仅 `LLM.generate` 方法略有差异：
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## 基准测试

基准测试代码参见 `bench.py`。

**测试配置：**
- 硬件：RTX 4070 Laptop（8GB）
- 模型：Qwen3-0.6B
- 总请求数：256 条序列
- 输入长度：在 100–1024 token 之间随机采样
- 输出长度：在 100–1024 token 之间随机采样

**性能结果：**
| 推理引擎 | 输出 Token 数 | 耗时（秒） | 吞吐量（tokens/s） |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)

---

> 本文件是仓库根目录 `README.md` 的中文翻译，代码块保留原文。
> 中文注解版源码与原理文档见 `doc_zh/` 目录。
