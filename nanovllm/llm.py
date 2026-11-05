# 导入真正的引擎实现；LLMEngine 里封装了进程管理、调度与执行的完整逻辑
from nanovllm.engine.llm_engine import LLMEngine


# LLM 是面向用户的“门面（facade）”类：直接继承 LLMEngine 且不增加任何成员。
# 单独存在只是为了对齐 vLLM 的 API 形态（vllm.LLM），把引擎细节藏在包内部。
class LLM(LLMEngine):
    pass
