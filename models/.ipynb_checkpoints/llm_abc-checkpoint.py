from abc import ABC, abstractmethod
from typing import Dict, Any, List

# --- LLM 和 Embedding 接口（保持不变）---
class AbstractLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """统一的模型生成接口。"""
        pass

class AbstractEmbedding(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """统一的文本向量化接口。"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """统一的查询向量化接口，用于查询嵌入。"""
        pass

# --- Tools 接口（保持不变）---
class AbstractTool(ABC):
    @abstractmethod
    def run(self, input_text: str) -> str:
        """统一的工具运行接口。"""
        pass

# --- Agent 接口（新增）---
class AbstractAgent(ABC):
    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agents的核心处理流程。
        接收当前的Agent状态（state），执行决策或业务逻辑，并返回更新后的状态。
        """
        pass

    @abstractmethod
    def get_agent_flow(self) -> Any:
        """
        返回 Agent 流程的 LangGraph 结构或可执行入口。
        """
        pass
