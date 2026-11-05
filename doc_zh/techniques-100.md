# Python 高级技法与重要库 100 讲

以 nano-vLLM（约 1300 行的 vLLM 复刻）真实代码为蓝本，讲解推理引擎实际用到的 100 个 Python 高级技法与重要库。每条含：技法/库名、分类、仓库位置、要点。行号对齐注释版源码。

**目录**：[Triton JIT](#一triton-jit-编程01-09) · [torch 编译与 CUDA 执行](#二torch-编译与-cuda-执行10-18) · [torch.distributed 与多进程](#三torchdistributed-与多进程19-29) · [flash_attn](#四flash_attn-与注意力30-35) · [张量与内存视图](#五张量与内存视图36-47) · [数值与随机](#六数值与随机48-52) · [数据结构与迭代协议](#七数据结构与迭代协议53-62) · [面向对象高级特性](#八面向对象高级特性63-72) · [函数式与控制流](#九函数式与控制流73-79) · [序列化与字节](#十序列化与字节80-85) · [类型注解与语言细节](#十一类型注解与语言细节86-93) · [工具库与工程实践](#十二工具库与工程实践94-100)

## 一、Triton JIT 编程（01-09）

**01. `@triton.jit` JIT 编译装饰器**
- 分类：Triton / GPU kernel
- 位置：`layers/attention.py`（`store_kvcache_kernel`，13 行）
- 要点：被装饰的 Python 函数不是普通函数，而是"内核模板"：首次按实参的 dtype/constexpr 组合编译成 PTX，之后复用。函数体里只能用 Triton 语言子集（`tl.*`），标量分支（`if slot == -1: return`）会在编译期展开。这是 nano-vLLM 唯一手写的 kernel：每个 program 拷贝一个 token 的整行 K/V 到分页 cache。

**02. `tl.constexpr` 编译期常量**
- 分类：Triton
- 位置：`layers/attention.py`（`D: tl.constexpr`，22 行）
- 要点：`D = num_heads * head_dim` 标成 constexpr 后成为编译期常量，`tl.arange(0, D)` 才能生成定长向量并让整行 load/store 完全展开。运行期才变的值只能当普通参数传，否则无法参与偏移向量的形状推导。

**03. `tl.program_id` 与一维 grid 并行模型**
- 分类：Triton
- 位置：`layers/attention.py`（24 行）
- 要点：`idx = tl.program_id(0)` 取当前 program 在 0 维 grid 里的编号；启动时 `[(N,)]` 表示开 N 个 program，正好一 token 一个。Triton 的编程模型是 SPMD：同一份代码在所有 program 上跑，靠 program_id 区分各自负责的数据分片，写起来像"单 token 的拷贝逻辑"，实际是 N 路并行。

**04. `tl.load` / `tl.store` 与指针算术**
- 分类：Triton
- 位置：`layers/attention.py`（26-31、39-40 行）
- 要点：Triton 没有张量对象，只有裸指针：`tl.load(ptr + offsets)` 按元素偏移向量一次性读出整行，`tl.store` 写回。偏移 = `行号*stride + tl.arange(0, D)`，把"二维寻址"压成一维算术。mask 参数可省（本例用早退代替），是极致简单的散射写。

**05. `tl.arange` 行内偏移向量**
- 分类：Triton
- 位置：`layers/attention.py`（28-31、37 行）
- 要点：`tl.arange(0, D)` 生成 `[0,1,...,D-1]` 的编译期向量，与标量起点相加得整行偏移。要求 D 是 2 的幂？不——`arange` 只要求编译期确定，本例 D 随模型头数变化，每个形状各自编译一份 kernel。

**06. 方括号 grid 启动语法**
- 分类：Triton
- 位置：`layers/attention.py`（58 行）
- 要点：`kernel[(N,)](args...)`：先以方括号传 grid（tuple，可多维），再像普通函数一样传参。JITFunction 的 `__getitem__` 返回一个"待启动"包装器，这是 Triton 的标志性 API 形态，与 CUDA 的 `<<<grid, block>>>` 对应。

**07. `Tensor.stride()` 与手工内存寻址**
- 分类：内存视图
- 位置：`layers/attention.py`（50-55 行）
- 要点：`stride(i)` 是第 i 维相邻元素间的元素数。kernel 只收首指针，Python 侧传 `key.stride(0)` 让 GPU 端自己算行起点——免 `.contiguous()` 拷贝。前提是布局符合假设（断言保证），这也是读别人 kernel 代码时最先要确认的事。

**08. 用 assert 建立"内存布局契约"**
- 分类：工程实践 / GPU
- 位置：`layers/attention.py`（50-55 行）
- 要点：四个 assert 分别锁定"最后一维连续、头间无 padding、cache 行宽恰为 D、token 数与 slot 数一致"。手写寻址的 kernel 一旦布局假设破裂就是静默写坏数据，把假设显式断言出来是 GPU 工程的防身术——宁可启动时崩，不要运行时错。

**09. 哨兵值 -1 与"图死批活"**
- 分类：Triton / CUDA Graph 协作
- 位置：`layers/attention.py`（27 行）；`engine/model_runner.py`（293-294 行）
- 要点：slot_mapping 里 -1 表示"这个位置不写 cache"，kernel 见 -1 直接返回。CUDA Graph 回放时桶尺寸向上取整，多余行全填 -1，让固定形状的图安全服务可变大小的批——一个哨兵值解决了图与动态批的根本矛盾。

## 二、torch 编译与 CUDA 执行（10-18）

**10. `@torch.compile` 局部编译**
- 分类：torch 2.x
- 位置：`layers/layernorm.py`（19、35 行）、`layers/rotary_embedding.py`（50 行）、`layers/activation.py`（8 行）、`layers/sampler.py`（8 行）
- 要点：把一串小算子（pow/mean/rsqrt/mul…）编译成一个融合 kernel，省 launch 与显存往返。nano-vLLM 只装饰"算子密集、形状规律"的小函数，不编译整模型——避免 flash_attn 外部算子和变长形状导致的反复重编译。装饰在方法上时首参 self 会进 trace，注意不要捕获 Python 可变态。

**11. 原位操作 `mul_` / `add_` / `div_`**
- 分类：torch 张量操作
- 位置：`layers/layernorm.py`（30、38 行）、`layers/sampler.py`（12、17 行）
- 要点：带下划线的方法原地修改自身，省一次新张量分配。在 compile 融合与 decode 高频路径（每步都跑）里，省分配就是省分配器与带宽开销。注意原位操作会破坏 autograd——推理引擎有 `inference_mode` 兜底才敢大量使用。

**12. `torch.inference_mode` 与 autograd 屏蔽**
- 分类：torch
- 位置：`engine/model_runner.py`（275、316 行）
- 要点：比 `no_grad` 更彻底：张量不带 version counter、不被记入 autograd 元数据，前向与"图捕获"都更快更省。推理引擎的所有前向都应包在它里面；vLLM 同样在 worker 层全局包。

**13. `torch.cuda.graph` 上下文管理器捕获**
- 分类：CUDA Graph
- 位置：`engine/model_runner.py`（345-347 行）
- 要点：`with torch.cuda.graph(g, pool):` 块内的 kernel 调用不真正执行，而是被录进 `CUDAGraph` 对象。捕获时张量地址被固化进图——所以输入输出必须预先分配在固定缓冲区。捕获前必须先 warmup（344 行），否则编译/分配事件会混进图里。

**14. `CUDAGraph.replay()` 一步重放**
- 分类：CUDA Graph
- 位置：`engine/model_runner.py`（301 行）
- 要点：`graph.replay()` 用一次 CPU 调用重放整段内核序列，decode 步骤数百个 kernel 的 launch 开销近乎归零。回放读的是"当初捕获时那些地址"里的**当前内容**——所以回放前把真实输入拷进固定缓冲区（288-299 行）就是全部接口。

**15. 跨图共享内存池 `graph.pool()`**
- 分类：CUDA Graph
- 位置：`engine/model_runner.py`（350 行）
- 要点：`with torch.cuda.graph(g, pool)` 传同一个 pool，让多张图的中间激活共用一块显存，避免每张图独立保留峰值。第一个（最大）图的池作为共享池，后续图在其上分配；销毁时整池一起释放。

**16. 桶形批尺寸（bucketed batch sizes）**
- 分类：工程模式
- 位置：`engine/model_runner.py`（333 行）
- 要点：图形状固化，不可能每个 bs 一张图。`[1,2,4,8] + range(16, 512, 16)` 让小批密集、大批稀疏，"向上取桶"的 padding 浪费与图数量之间取平衡。vLLM 的 cudagraph capture sizes 同款思路。

**17. 固定地址输入/输出缓冲区（graph_vars）**
- 分类：CUDA Graph
- 位置：`engine/model_runner.py`（325-330、355-362 行）
- 要点：捕获前 `torch.zeros` 预分配 input_ids/positions/slot_mapping/context_lens/block_tables/outputs 六个缓冲区并保存在 `graph_vars`；回放前 `[:bs]` 切片原地覆写。多余行填 -1/0 屏蔽（293-298 行）。"图是死的、数据是活的"全靠这套缓冲区协议。

**18. warmup-then-capture 纪律**
- 分类：CUDA Graph / 工程实践
- 位置：`engine/model_runner.py`（344-347 行）
- 要点：每张图捕获前先在普通流上跑一遍同样的前向：cuBLAS 句柄、workspace、懒加载的 kernel 都就位后，捕获流里才不会出现"首次分配/编译"这类不能被录制的操作。跳过 warmup 是 CUDA Graph 最常见的坑。

## 三、torch.distributed 与多进程（19-29）

**19. `dist.init_process_group` 与 TCP rendezvous**
- 分类：torch.distributed
- 位置：`engine/model_runner.py`（31 行）
- 要点：`init_process_group("nccl", "tcp://localhost:2333", world_size, rank)`：后端选 NCCL（GPU 张量集合通信），用固定端口的 TCP 让各 rank 互相发现。单机多卡的极简 bootstrapping——vLLM 有更复杂的 store/环境变量机制，原理相同。

**20. `dist.all_reduce` 求和聚合**
- 分类：torch.distributed / TP
- 位置：`layers/linear.py`（199 行）、`layers/embed_head.py`（53 行）
- 要点：行并行的输出与词表并行的 embedding 都靠 all_reduce(SUM) 把各 rank 的部分和加成完整结果，所有 rank 得到同一份。它是 TP 每层固定的一次通信成本，也决定了"列并行→行并行"配对只需每子层一次通信。

**21. `dist.gather` 定向收集**
- 分类：torch.distributed / TP
- 位置：`layers/embed_head.py`（84-86 行）
- 要点：LM head 输出的词表分段 logits 用 gather 聚到 rank 0 再 `torch.cat` 拼接——只有采样方（rank 0）需要完整 logits 时，gather 比 all_gather/all_reduce 通信量更小。接收列表只在目标 rank 上非 None。

**22. `dist.barrier` 进程组栅栏**
- 分类：torch.distributed
- 位置：`engine/model_runner.py`（59-62 行）
- 要点：rank 0 创建共享内存、其他 rank 打开——中间插 barrier 保证"创建先于打开"。栅栏不传数据，只传"大家都到了这一步"的同步语义，是初始化顺序依赖的标准解法。

**23. `mp.get_context("spawn")`**
- 分类：multiprocessing
- 位置：`engine/llm_engine.py`（32 行）
- 要点：CUDA 上下文不能被 fork 出的子进程继承（会死锁/崩溃），spawn 通过重新 import 主模块在子进程里干净地重建一切。显式取 spawn context 再建 Process，避免依赖平台默认（Linux 默认 fork）。

**24. `mp.Process` 以构造函数为入口**
- 分类：multiprocessing
- 位置：`engine/llm_engine.py`（37 行）
- 要点：`target=ModelRunner, args=(config, i, event)`——子进程入口直接是执行器的 `__init__`，初始化完（建模型、捕图）顺理成章进入常驻 `loop()`。省掉"先定义 worker 函数再传参"的样板，对象生命周期与进程生命周期天然一致。

**25. `multiprocessing.Event` 门铃模式**
- 分类：multiprocessing 同步
- 位置：`engine/llm_engine.py`（35-37 行）、`engine/model_runner.py`（91-101、112-113 行）
- 要点：主 rank 写完信箱后 `event.set()` 广播，子进程 `event.wait()` 阻塞等铃、读完 `event.clear()` 复位。Event 是跨进程的布尔旗标，比轮询省 CPU、比 Pipe 轻——典型的"生产者写信 + 按铃"信箱模型。

**26. `multiprocessing.shared_memory.SharedMemory`**
- 分类：IPC
- 位置：`engine/model_runner.py`（57-62 行）
- 要点：rank 0 `create=True` 建 1MB 段，子进程按 name 附属打开。POSIX 共享内存在无拷贝的前提下让多个进程看到同一块字节——用来传"方法调用"这种小而高频的控制消息正合适。清理要 close+unlink（67-72 行），unlink 只由创建者做。

**27. memoryview 切片 + 定长头协议**
- 分类：IPC / 字节协议
- 位置：`engine/model_runner.py`（94-95、108-109 行）
- 要点：`shm.buf` 是 memoryview：`buf[0:4]` 放小端 4 字节长度头，`buf[4:n+4]` 放 payload。读者先读长度再截取，写者先写长度后写内容。1MB 上限即"信箱容量"，payload 超限会崩——所以 `Sequence.__getstate__` 才要精简（见 80）。

**28. `pickle.dumps/loads` 实现迷你 RPC**
- 分类：序列化 / IPC
- 位置：`engine/model_runner.py`（95、107 行）
- 要点：`(方法名, *args)` 整体 pickle 后过共享内存，子进程 unpickle 后 `getattr(self, method_name)(*args)` 反射调用（115-119 行）。对象方法 + pickle + 共享内存 = 20 行的多进程 RPC。张量大数据不走这条路，走 NCCL 集合通信。

**29. 全局默认 dtype/device 切换与恢复**
- 分类：torch 全局状态
- 位置：`engine/model_runner.py`（34-39、45-46 行）
- 要点：`torch.set_default_dtype(hf_config.dtype)` + `set_default_device("cuda")` 让后续 `nn.Parameter(torch.empty(...))` 直接按目标 dtype 建在 GPU 上，免去逐参数 `.to()`。用完恢复原值——全局状态是借的，不是改的。

## 四、flash_attn 与注意力（30-35）

**30. `flash_attn_varlen_func` 变长批接口**
- 分类：flash_attn
- 位置：`layers/attention.py`（98-102 行）
- 要点：prefill 批内序列长短不一，varlen 接口免 padding：所有 token 扁平拼成 `[Σtokens, heads, dim]`，用 `cu_seqlens_q/k`（前缀和）划界、`max_seqlen_q/k` 供调度。一次调用算完整个混合批，是吞吐的关键。

**31. `cu_seqlens` 累积长度约定**
- 分类：flash_attn / 数据格式
- 位置：`engine/model_runner.py`（185-186、202-203 行）
- 要点：`cu_seqlens = [0, len1, len1+len2, ...]`，第 i 条序列占扁平张量的 `cu[i]:cu[i+1]`。Python 侧用 `append(cu[-1] + seqlen)` 逐条累加生成。同一约定也被 LM head 复用：`cu_seqlens_q[1:]-1` 就是每条序列末 token 的下标（embed_head.py:76）。

**32. `flash_attn_with_kvcache` 解码接口**
- 分类：flash_attn
- 位置：`layers/attention.py`（105-108 行）
- 要点：decode 时 q 只有 1 个 token/序列（`unsqueeze(1)` 补 seq 维），K/V 直接给整块 cache 张量，kernel 按 `cache_seqlens` 与 `block_table` 跳页读取。它是 flash_attn 内建的 paged attention 支持——引擎自己不用写注意力 kernel。

**33. `block_table` 分页读取**
- 分类：flash_attn / PagedAttention
- 位置：`layers/attention.py`（100、107 行）；`engine/model_runner.py`（171-176 行）
- 要点：block_table 是 `[batch, max_blocks]` 的 int32 页表，-1 补齐到等长（175 行）。kernel 拿逻辑块号查表得物理块号，实现"逻辑连续、物理离散"。prefix cache 命中时 prefill 也走同一条路（attention.py:93-95）。

**34. GQA 头数不对称广播**
- 分类：注意力结构
- 位置：`models/qwen3.py`（32-38 行）、`layers/attention.py`（61-81 行）
- 要点：Q 头多、KV 头少（如 32:8），flash_attn 原生支持按组广播。TP 下每 rank 分到 `num_heads//tp` 个 Q 头与 `num_kv_heads//tp` 个 KV 头，q_size/kv_size 记录各自宽度供 split 用。

**35. `softmax_scale` 与 causal 掩码**
- 分类：注意力数学
- 位置：`models/qwen3.py`（39 行）、`layers/attention.py`（101、107 行）
- 要点：`scale = head_dim ** -0.5` 抵消点积随维度增长的方差；`causal=True` 让 token 只看前文。这两个参数跟着调用传给 flash_attn，引擎本身不实现注意力数学。

## 五、张量与内存视图（36-47）

**36. `Tensor.view` 零拷贝重解释形状**
- 分类：torch 视图
- 位置：`models/qwen3.py`（95-99 行）
- 要点：`q.view(-1, num_heads, head_dim)` 把 `[tokens, q_size]` 拆成 `[tokens, heads, dim]`，不移动数据。view 要求内存连续兼容——这正是 TP 线性层把输出排成 Q|K|V 并排的原因之一。

**37. `torch.chunk` 等分切块**
- 分类：torch 视图
- 位置：`models/qwen3.py`（93 行）、`layers/activation.py`（13 行）、`layers/rotary_embedding.py`（11、56 行）
- 要点：`chunk(n, dim)` 沿维度切成 n 份：QKV split 前的多路解包、SwiGLU 的 gate/value 两半、RoPE 的 x1/x2 两半都用它。返回视图，切分零拷贝。

**38. `Tensor.split` 按尺寸列表切分**
- 分类：torch 视图
- 位置：`models/qwen3.py`（93 行）
- 要点：`qkv.split([q_size, kv_size, kv_size], dim=-1)` 按给定宽度切开不等长的段——GQA 下 Q 段宽、K/V 段窄，chunk 做不到，split 可以。与 chunk 一样返回视图。

**39. `Tensor.narrow` 定位切片**
- 分类：torch 视图
- 位置：`layers/linear.py`（84-88、109-120、155-160 行）、`layers/embed_head.py`（33-36 行）
- 要点：`narrow(dim, start, length)` 等价于切片语法但可用变量表达，是 weight_loader 的主力：从完整 HF 权重里取出"本 rank 分片"再 `copy_` 进切分后的参数。全程视图，最后才一次拷贝。

**40. `flatten` / `unsqueeze` 维度整形**
- 分类：torch 视图
- 位置：`models/qwen3.py`（108 行）、`layers/sampler.py`（12 行）、`layers/embed_head.py`（51 行）
- 要点：`o.flatten(1, -1)` 把 `[tokens, heads, dim]` 压回 `[tokens, heads*dim]` 交给 o_proj；`temperatures.unsqueeze(1)` 升维以广播除法。一压一升是 batch 维与特征维之间的日常搬运。

**41. 高级索引取行 + `contiguous()`**
- 分类：torch 索引
- 位置：`layers/embed_head.py`（76-78 行）
- 要点：`x[last_indices]` 用下标张量取"每条序列最后一个 token"的 hidden，返回新张量（有拷贝）；紧接 `.contiguous()` 保证后续 `F.linear` 走最快的连续路径。高级索引是 gather 语义，布尔掩码是 scatter 语义的镜像。

**42. 布尔掩码乘法实现"段外清零"**
- 分类：torch 索引 / TP
- 位置：`layers/embed_head.py`（43-51 行）
- 要点：词表并行时每个 rank 只"认领"落在自己区间的 id：`mask * (x - start)` 把段外 id 归零查询，`mask.unsqueeze(1) * y` 把段外结果清零，all_reduce 后恰好只剩正确值。用算术代替分支，全张量化。

**43. `pin_memory=True` + `non_blocking=True` 异步 H2D**
- 分类：CUDA 传输
- 位置：`engine/model_runner.py`（175、233-237、256-259 行）
- 要点：锁页内存允许 DMA 直接传输；`.cuda(non_blocking=True)` 提交后立即返回，CPU 继续打包、GPU 异步接收。所有每步重建的小张量都走这条路，把传输延迟藏进计算。

**44. `torch.empty` / `torch.zeros` 预分配**
- 分类：torch 显存管理
- 位置：`engine/model_runner.py`（161、325-330 行）
- 要点：KV cache 大张量与 graph 缓冲区都"启动时一次分配、运行期零申请"。`empty` 不初始化（KV 反正会被覆写），`zeros` 用于需要确定初值的 graph 缓冲区。运行期避免任何分配是推理引擎稳定性的底线。

**45. `register_buffer(persistent=False)`**
- 分类：nn.Module 状态管理
- 位置：`layers/rotary_embedding.py`（48 行）
- 要点：cos_sin_cache 是"随模型走但不随梯度走"的数据：`.to(device)` 会跟着搬、state_dict 里却不保存（可由超参重建）。persistent=False 避免把 80KB×2 的表写进 checkpoint，也避免 load 时形状对不上。

**46. `nn.Parameter` 与 `register_parameter("bias", None)`**
- 分类：nn.Module
- 位置：`layers/linear.py`（29、38-42 行）
- 要点：`nn.Parameter(torch.empty(...))` 把张量注册为可学习参数并进入 `_parameters`；bias 不存在时显式注册 None（而不是不写属性），让 `state_dict()`、`named_parameters()` 与 `F.linear(x, w, None)` 的行为都确定。

**47. 不定长列表补 -1 对齐成矩阵**
- 分类：工程模式
- 位置：`engine/model_runner.py`（174-175 行）
- 要点：`block_table + [-1] * (max_len - len(...))` 把批内长短不一的页表补成规整矩阵——GPU 端需要矩形张量，-1 由下游语义（无效页）消化。同类手法：graph 缓冲区补 0、slot 补 -1。

## 六、数值与随机（48-52）

**48. RMSNorm 的 rsqrt 融合写法**
- 分类：数值计算
- 位置：`layers/layernorm.py`（28-31 行）
- 要点：`x * rsqrt(mean(x²)+eps)`：`pow(2).mean(dim=-1, keepdim=True)` 求均方、`rsqrt` 一次算出倒数平方根（比 1/sqrt 快且稳），`mul_` 原位缩放。keepdim 保证广播对齐；整个表达式被 torch.compile 融合成单 kernel。

**49. `torch.einsum` 外积建相位表**
- 分类：torch / 张量代数
- 位置：`layers/rotary_embedding.py`（40 行）
- 要点：`einsum("i,j -> ij", t, inv_freq)` 一步算出 `位置 × 频率` 网格，替代 `t[:, None] * inv_freq[None, :]`。einsum 的记号法在读复杂张量收缩（谁乘谁、哪些维消掉）时比手写下标可读得多。

**50. RoPE 的"前后对半"旋转（GPT-NeoX 风格）**
- 分类：数值算法
- 位置：`layers/rotary_embedding.py`（6-16 行）
- 要点：把头维劈成两半 (x1, x2)，按二维旋转矩阵 `y1 = x1·cos − x2·sin, y2 = x2·cos + x1·sin` 旋转。与 HF 的 rotate_half（交错劈法）数值等价但排布更简单。旋转在 fp32 里算（11 行 `.float()`），算完转回 bf16。

**51. 指数竞赛采样（Gumbel 变体）**
- 分类：随机算法
- 位置：`layers/sampler.py`（17 行）
- 要点：`argmax(probs / Exp(1) 噪声)` 数学上等价于按 probs 多项式采样一次（Gumbel-max 的指数形式）。相比 `torch.multinomial` 免去前缀和与规约，纯逐元素运算，能被 torch.compile 融合且无同步点——采样最快的写法之一。

**52. `exponential_` / `clamp_min_`**
- 分类：torch 随机
- 位置：`layers/sampler.py`（17 行）
- 要点：`torch.empty_like(probs).exponential_(1)` 原地填充标准指数噪声；`clamp_min_(1e-10)` 防止除零产生 inf/NaN。随机数直接在 GPU 上生成，不过 CPU、无需 seed 同步（每个 rank 独立噪声也只影响 rank 0 的采样）。

## 七、数据结构与迭代协议（53-62）

**53. `collections.deque` 双端队列**
- 分类：数据结构
- 位置：`engine/scheduler.py`（20-22 行）、`engine/block_manager.py`（46 行）
- 要点：deque 两端进出都是 O(1)，list 的 `pop(0)` 是 O(n)。调度器的 waiting/running 队列、块管理器的空闲块池全用它——"队列"在 Python 里永远该用 deque。

**54. `popleft` / `appendleft` 队列与插队**
- 分类：数据结构
- 位置：`engine/scheduler.py`（74、118 行）
- 要点：FIFO 主体是 `popleft()` 取队首；抢占回来的序列 `appendleft()` 插到队首优先重算。同一个数据结构同时提供"排队"与"插队"两种语义，是抢占式调度的天然载体。

**55. `extendleft(reversed(...))` 保序批量回填**
- 分类：Python 惯用法
- 位置：`engine/scheduler.py`（108 行）
- 要点：`extendleft` 会把可迭代对象**逆序**逐个 `appendleft`——想按原顺序放回队首就得先 `reversed` 一次。这是 deque 最著名的反直觉点，记不住就会悄悄把队列顺序搞反。

**56. `pop()` 把 deque 当栈：抢占牺牲者**
- 分类：数据结构
- 位置：`engine/scheduler.py`（92 行）
- 要点：缺块时 `self.running.pop()` 从**队尾**拿牺牲者（最晚进队、进度最浅的），队首老序列得以保留。同一容器的 popleft/pop 分别充当 FIFO 与 LIFO，一行完成"选谁去死"的策略。

**57. `itertools.count` 无限发号器**
- 分类：迭代协议
- 位置：`engine/sequence.py`（21、25 行）
- 要点：类属性 `counter = count()` 是进程级无限计数器，`next(Sequence.counter)` 领全局唯一 seq_id。比 `self._n += 1` 优雅：无锁（GIL 下 next 原子）、无状态字段、所有实例天然共享。

**58. 生成器表达式 + `sum` / `max` / `min`**
- 分类：迭代协议
- 位置：`engine/llm_engine.py`（73 行）、`engine/model_runner.py`（172 行）、`engine/scheduler.py`（41 行）
- 要点：`sum(seq.num_scheduled_tokens for seq in seqs)`、`max(len(seq.block_table) for seq in seqs)`——惰性生成器喂聚合函数，不建中间列表。写引擎时到处是"对这批序列聚合某个量"，这是最顺手的句式。

**59. `next(gen)` 取首个匹配**
- 分类：迭代协议
- 位置：`engine/model_runner.py`（286 行）
- 要点：`next(x for x in self.graph_bs if x >= bs)` 在桶列表里找"≥bs 的最小桶"，等价于二分但代码只有一行。生成器惰性求值，找到即停。

**60. 列表/字典推导式**
- 分类：语言特性
- 位置：`engine/llm_engine.py`（22、78、122、124 行）、`engine/model_runner.py`（174-175 行）
- 要点：`{k: v for k, v in kwargs.items() if k in config_fields}` 过滤配置、`[seq.block_table + [-1] * ...]` 对齐页表、`[(seq.seq_id, ...) for seq in seqs if seq.is_finished]` 收集结果。推导式是 Python 的"声明式集合变换"，比等价 for 循环少一半行数。

**61. `zip` 平行遍历**
- 分类：迭代协议
- 位置：`engine/scheduler.py`（122 行）、`engine/llm_engine.py`（103 行）
- 要点：`for seq, token_id in zip(seqs, token_ids)` 把"序列"与"它采样出的 token"一一配对——两个列表顺序由上游约定保证。引擎里"成对推进"的逻辑（调度结果 × 采样结果）都靠 zip。

**62. 字典收集 + `sorted(keys)` 还原顺序**
- 分类：工程模式
- 位置：`engine/llm_engine.py`（106、119-124 行）
- 要点：完成的序列以 `outputs[seq_id] = token_ids` 进字典（完成顺序≠请求顺序），最后 `sorted(outputs.keys())` 按 seq_id 排序还原与 prompts 一一对应的输出。乱序完成、有序交付，是并发系统的通用收尾模式。

## 八、面向对象高级特性（63-72）

**63. `@dataclass(slots=True)`**
- 分类：语言特性
- 位置：`config.py`（8 行）、`utils/context.py`（9 行）、`sampling_params.py`（5 行）
- 要点：dataclass 免写 `__init__`；`slots=True` 同时生成 `__slots__`。适合"纯字段载体"类：Config、SamplingParams、Context 都是没有行为负担的值对象。

**64. `__slots__` 实例字段固定**
- 分类：语言特性
- 位置：同上三个类
- 要点：slots 让实例不再有 `__dict__`：属性偏移化访问更快、内存更小、还顺带禁止动态加属性（拼错字段名立刻 AttributeError，等于免费的白名单校验）。Context 每步新建一个，slots 的分配开销优势正好用在该刀刃上。

**65. `dataclasses.fields` 内省过滤 kwargs**
- 分类：元编程
- 位置：`engine/llm_engine.py`（22-23 行）
- 要点：`{f.name for f in fields(Config)}` 拿到配置类字段名集合，再 `{k: v for k, v in kwargs.items() if k in config_fields}` 只放行认识的参数。用户传错参数不炸引擎——用反射做"宽容的构造函数"，vLLM 的 EngineArgs 同思路。

**66. `__post_init__` 初始化钩子**
- 分类：dataclass
- 位置：`config.py`（34-43 行）、`sampling_params.py`（15-17 行）
- 要点：dataclass 生成 `__init__` 后自动调用 `__post_init__`，是唯一能安插"校验 + 派生字段"的位置：Config 在这里读 HF 配置、取 max_model_len 上界；SamplingParams 在这里拒绝 temperature≤0。

**67. `Enum` + `auto()`**
- 分类：语言特性
- 位置：`engine/sequence.py`（9-13 行）
- 要点：`SequenceStatus` 三态用 Enum 而非字符串/整数：成员有身份（`is` 比较）、拼错即报错、可迭代。`auto()` 免去手工编号。状态机的状态字段永远是 Enum 的用武之地。

**68. `@property` 派生只读字段**
- 分类：语言特性
- 位置：`engine/sequence.py`（64-88 行）
- 要点：`is_finished`、`num_completion_tokens`、`num_blocks`、`last_block_num_tokens` 都是"能从已有字段算出来"的量——用 property 现算现取，杜绝冗余状态不同步。调用侧写起来像字段，实际是方法。

**69. 双下划线协议：`__len__` / `__getitem__`**
- 分类：语言特性
- 位置：`engine/sequence.py`（50-57 行）
- 要点：实现 `__len__` 后 `len(seq)` 返回 token 数；实现 `__getitem__` 后 `seq[start:end]` 直接切片 token 列表。Sequence 借此"伪装"成一个 token 序列，调度器/打包代码读起来自然。这是 Python duck typing 的精髓：实现协议即获得语法。

**70. 函数挂为对象属性：weight_loader 策略位**
- 分类：OOP 技巧
- 位置：`layers/linear.py`（32 行）、`utils/loader.py`（28-33 行）
- 要点：`self.weight.weight_loader = self.weight_loader` 把"如何把 HF 权重装进这个参数"的知识直接挂在参数对象上；loader 统一 `getattr(param, "weight_loader", default)` 分派。等效策略模式，但不需要类继承——每个参数自带加载策略，新增并行层零改动 loader。

**71. `nn.Module.__getattr__` 与参数注册机制**
- 分类：元编程 / PyTorch
- 位置：`layers/linear.py`、`models/qwen3.py`（全部子模块）、`engine/model_runner.py`（164-170 行）
- 要点：`self.qkv_proj = ...` 触发 nn.Module 的 `__setattr__` 把子模块/参数登记进 `_modules/_parameters` 字典；`model.modules()` 遍历、`get_parameter("model.layers.0...")` 按名寻址全靠这套注册表。理解它才能理解"属性赋值即建模"和 loader 的按名加载。

**72. 类属性做进程级共享配置**
- 分类：OOP
- 位置：`engine/sequence.py`（18-21 行）
- 要点：`block_size`/`counter` 挂在 Sequence 类上而非实例：所有序列共享同一 block_size，且 pickle 传输实例时类属性不会跟着走（子进程反序列化后自动用自己模块里的类属性），天然规避"配置重复传输"。

## 九、函数式与控制流（73-79）

**73. `functools.lru_cache` 做工厂缓存**
- 分类：函数式
- 位置：`layers/rotary_embedding.py`（67-75 行）
- 要点：`@lru_cache(1)` 装饰 `get_rope(head_size, ...)`：参数相同的多层 Attention 拿到**同一个** RoPE 实例（省显存也省 cos/sin 表重复建）。lru_cache 的一般套路——"按参数记忆化的构造函数"，比手写单例字典干净。

**74. `for-else`：循环未 break 即执行**
- 分类：语言特性
- 位置：`utils/loader.py`（33-37 行）
- 要点：内层 `for k in mapping: if k in weight_name: ...; break` 的 else 分支表示"没有任何映射命中"——普通权重走默认路径。for-else 专治"循环里找了半天没找到"的语义，比加 found 标志变量清爽。

**75. `while-else`：抢占循环的出口区分**
- 分类：语言特性
- 位置：`engine/scheduler.py`（87-104 行）
- 要点：`while not can_append: ... preempt ...` 的 else 只在"循环条件自然变假"（显存够了）时执行——即正常调度路径；`break`（自己也被抢占）则跳过 else。一个 else 把"成功"与"放弃"两条出口写在一处，不需要 success 标志。

**76. `atexit` 注册进程退出清理**
- 分类：标准库
- 位置：`engine/llm_engine.py`（49 行）
- 要点：`atexit.register(self.exit)` 保证 TP 子进程在解释器退出时被 join，不留僵尸进程占着显卡。清理逻辑与构造逻辑放同一个类里注册，不依赖调用方记得调 close。

**77. `copy()` 防御性浅拷贝**
- 分类：工程习惯
- 位置：`engine/sequence.py`（28 行）
- 要点：`self.token_ids = copy(token_ids)` 复制传入列表再持有——引擎内部会 append，若直接引用外部列表会互相污染。浅拷贝足够：元素是 int，不可变。

**78. `global` 语句与模块级单例**
- 分类：语言特性
- 位置：`utils/context.py`（27-41 行）
- 要点：模块级 `_CONTEXT` 变量配 `global` 声明的 setter/getter/reset 三函数，构成最朴素的进程级单例。整体替换实例（而非改字段）保证单步原子切换。跨层传参的替代方案，代价是隐式依赖——所以 reset_context 的调用纪律很重要。

**79. `getattr(obj, name, default)` 容错读取**
- 分类：语言特性
- 位置：`models/qwen3.py`（155-158 行）、`utils/loader.py`（16、33 行）、`engine/model_runner.py`（152 行）
- 要点：不同版本 Qwen3Config 的可选字段用 `getattr(config, "head_dim", None)` 兜底；loader 用 `getattr(param, "weight_loader", default)` 区分"有策略"与"用默认"。三参 getattr 是"可能有、可能没有"场景的标准答案。

## 十、序列化与字节（80-85）

**80. pickle 协议定制：`__getstate__` / `__setstate__`**
- 分类：序列化
- 位置：`engine/sequence.py`（103-119 行）
- 要点：默认 pickle 会搬整个对象图；自定义后 decode 序列只传 `(长度们, block_table, last_token)` 六元组——不传完整 token_ids 列表。TP 多卡时这是每步、每序列都要过的 IPC 通道，协议瘦身直接减通信字节。`__setstate__` 里按 last_state 的类型（list vs int）区分 prefill/decode 两种还原路径。

**81. `int.to_bytes` / `int.from_bytes` 定长头**
- 分类：字节协议
- 位置：`engine/model_runner.py`（94、108 行）
- 要点：`n.to_bytes(4, "little")` 把长度编码成 4 字节小端；读端 `int.from_bytes(buf[0:4], "little")` 还原。比 struct.pack 更轻的定长整数编解码，适合自家人之间的私有协议。

**82. `xxhash.xxh64` 非加密快速哈希**
- 分类：第三方库
- 位置：`engine/block_manager.py`（53-63 行）
- 要点：prefix cache 要对"每个 block 的 token 序列"算哈希，性能敏感但不需要密码学强度——xxhash 比 hashlib(md5/sha) 快一个数量级。`intdigest()` 直接给 64 位整数，正好做 dict key 与链式哈希的输入。

**83. `numpy.array(...).tobytes()` 连续字节流**
- 分类：序列化
- 位置：`engine/block_manager.py`（60 行）
- 要点：哈希喂的是字节；`np.array(token_ids).tobytes()` 把 id 列表压成紧凑连续字节（比 `b"".join` 或 `struct.pack` 快且短）。约定"同列表同字节"，哈希才稳定。

**84. 链式前缀哈希原理**
- 分类：算法设计
- 位置：`engine/block_manager.py`（51-63、164-179 行）
- 要点：`hash_i = H(hash_{i-1} ‖ tokens_i)` 把父块哈希掺进子块——前缀任何一块不同，之后所有块哈希全变。于是"哈希相等 ⟺ 整条前缀逐块相等"，一次 dict 查找即完成最长公共前缀匹配，无须比较序列本身。与 vLLM 的 block hash（parent hash + token ids）完全同构。

**85. `safetensors.safe_open` 惰性权重加载**
- 分类：第三方库
- 位置：`utils/loader.py`（20-37 行）
- 要点：`safe_open(file, "pt", "cpu")` 上下文管理器只读元数据；`f.keys()` 列名、`f.get_tensor(name)` 时才真正读那块数据。大模型分片逐个打开、逐个取张量，峰值内存只多一份权重，不做全量反序列化。

## 十一、类型注解与语言细节（86-93）

**86. PEP 604 联合类型 `X | None`**
- 分类：类型注解
- 位置：`config.py`（25 行）、`utils/context.py`（12-19 行）、`layers/linear.py`（21 行）等
- 要点：`int | None` 取代 `Optional[int]`（3.10+），`str | list[int]` 表达"多态入参"（llm_engine.py:60）。注解虽不参与运行时（除 dataclass 字段），却是读代码时最高密度的文档。

**87. 内置泛型 `list[int]` / `tuple[X, bool]` / `dict[int, int]`**
- 分类：类型注解
- 位置：`engine/scheduler.py`（35 行）、`engine/block_manager.py`（42-46 行）等
- 要点：3.9+ 可直接用内置容器做注解，`schedule() -> tuple[list[Sequence], bool]` 把"返回序列列表 + 是否 prefill"的契约写在签名上。配合 deque 等泛型参数（`deque[int]`），IDE 与读者双赢。

**88. `*args` 星号收集与解包**
- 分类：语言特性
- 位置：`engine/model_runner.py`（91-119 行）
- 要点：`method_name, *args = pickle.loads(...)` 把调用拆成名字+参数包；`self.call(method_name, *args)` 再原样展开转发。一收一发实现"调用透传"，是 RPC/代理层的标配句式。

**89. 元组多重赋值解包**
- 分类：语言特性
- 位置：`engine/sequence.py`（110-111 行）、`layers/activation.py`（13 行）
- 要点：`a, b, c, d, e, f = state` 按位置还原 pickle 元组；`x, y = x.chunk(2, -1)` 接住切分结果。位置约定要两端一致——这也是为什么 getstate/setstate 必须成对出现、注释对齐。

**90. `isinstance` 运行时分派**
- 分类：语言特性
- 位置：`engine/llm_engine.py`（62 行）、`engine/sequence.py`（111 行）、`engine/llm_engine.py`（101 行）、`models/qwen3.py`（63 行）
- 要点：prompt 是字符串就分词、是列表就直用；setstate 里 last_state 是 list 还是 int 决定还原路径；rope_scaling 是不是 dict 决定读不读子字段。鸭子类型之上的显式分派，比全靠约定稳。

**91. `assert` 作为不变量契约**
- 分类：工程实践
- 位置：`config.py`（35-39 行）、`engine/scheduler.py`（105 行）、`engine/block_manager.py`（67 行）等
- 要点：整数除尽、块数>0、decode 批非空、空闲块引用为 0……引擎在每条"数学上应成立"的路径上放 assert。它既是运行期护栏又是文档：读到 assert 就知道此处依赖什么前提。（注意 assert 会被 -O 优化掉，生产引擎要用显式异常——教学代码可接受。）

**92. 条件表达式（三元）**
- 分类：语言特性
- 位置：`engine/model_runner.py`（280、308、312 行）、`layers/linear.py`（196 行）
- 要点：`a if cond else b` 内联小分派：prefill/decode 的打包函数选择、`self.bias if self.tp_rank == 0 else None` 的 bias 归属、rank 0 才采样。让"两种取值二选一"不占用一个 if 块的视觉空间。

**93. 向上取整除法 `(n + b - 1) // b`**
- 分类：整数运算惯用法
- 位置：`engine/sequence.py`（83 行）、`engine/model_runner.py`（214、321 行）
- 要点：num_blocks、end_block、max_num_blocks 全靠它：不用 math.ceil（避免 float 精度问题）、不用 `-(-n//b)` 的花活。分页系统的第一公式——"n 个 token 要几页"。

## 十二、工具库与工程实践（94-100）

**94. `tqdm` 进度条与 `set_postfix`**
- 分类：第三方库
- 位置：`engine/llm_engine.py`（92、112-115 行）
- 要点：`tqdm.auto` 自动适配 notebook/终端；`set_postfix({...})` 每步刷新 Prefill/Decode 两条吞吐；`disable=not use_tqdm` 让 benchmark 静默。进度更新用 `pbar.update(1)` 按完成的序列数推进，而不是按步数。

**95. `time.perf_counter` 高精度计时**
- 分类：标准库
- 位置：`engine/llm_engine.py`（109-114 行）
- 要点：测吞吐要用单调高精度钟，`perf_counter` 比 `time.time` 分辨率高且不受系统对时影响。每步计时算瞬时吞吐，prefill/decode 用 num_tokens 的正负号区分记账（73 行）。

**96. `transformers.AutoConfig` / `AutoTokenizer`**
- 分类：第三方库（HF）
- 位置：`config.py`（26 行）、`engine/llm_engine.py`（44 行）、`models/qwen3.py`（4 行）
- 要点：模型结构参数（层数/头数/dtype）与 tokenizer 都从 HF 生态借——nano-vLLM 复刻的是推理引擎，不是模型定义生态。`AutoConfig.from_pretrained` 读出的 hf_config 一路传到各层；`config.eos` 用 tokenizer 的 eos 覆写（llm_engine.py:46）。

**97. `glob` + `os.path` 文件遍历**
- 分类：标准库
- 位置：`utils/loader.py`（14 行）、`config.py`（35 行）
- 要点：`glob(os.path.join(path, "*.safetensors"))` 找齐所有分片文件（大模型权重常拆多文件）。`os.path.isdir` 断言模型目录存在——fail fast 在入口处。

**98. `torch.cuda.mem_get_info` / `memory_stats` 显存预算**
- 分类：torch CUDA
- 位置：`engine/model_runner.py`（145-157 行）
- 要点：`mem_get_info()` 给驱动视角的 free/total（含其他进程占用），`memory_stats()` 给分配器视角的 peak/current——两者相减取"最坏情况下真可用于 KV cache 的量"再除以块字节数。vLLM 的 profile-run 测量同此原理。

**99. 显存管理三件套：`empty_cache` / `reset_peak_memory_stats` / `synchronize`**
- 分类：torch CUDA
- 位置：`engine/model_runner.py`（125-126、137、351、358-359 行）
- 要点：warmup 前清缓存并重置峰值统计（测量才准）、warmup 后归还缓存、捕图后 synchronize 确保录制完整。这三板斧出现在每个"测量/捕获"节点——分配器的统计只反映 PyTorch 自己，驱动层还有缓存池要手动清。

**100. 基准测试方法学**
- 分类：工程实践
- 位置：`bench.py`（1-32 行）
- 要点：控制变量做对照：同 seed 生成同分布负载（17 行）、`ignore_eos=True` 保证输出长度可控（18 行）、先跑一次小 `generate` 预热再计时（22-24 行）、吞吐 = `sum(max_tokens)/耗时`（26-28 行）。还留了一行注释展示如何换成 vLLM 跑同样负载——可复现对比是性能工作的起点。
