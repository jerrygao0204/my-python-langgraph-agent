from typing import Dict, Type
# 从 llm_factory 导入 BaseFactory 和 AbstractEmbedding
from factory.llm_factory import BaseFactory 
from models.llm_abc import AbstractEmbedding 
from models.implementations import OpenAIEmbeddingsModel, HuggingFaceEmbeddingsModel # 导入具体实现

# 注册表：将配置中的 provider 映射到实际的 Python 类
EMBEDDING_MAP: Dict[str, Type[AbstractEmbedding]] = {
    "openai": OpenAIEmbeddingsModel,
    "huggingface": HuggingFaceEmbeddingsModel,
}

class EmbeddingFactory(BaseFactory):
    """Embedding Factory，继承自 BaseFactory，负责创建 Embedding 实例。"""
    def get_instance(self, component_key: str) -> AbstractEmbedding:
        """
        component_key: 配置中 embedding 部分的键名，如 'text_embedding'。
        """
        config_key = "embedding"
        component_config, EmbeddingClass = self._get_config_and_class(config_key, component_key, EMBEDDING_MAP)
        
        # 实例化并返回 Embedding 对象
        print(f"\n--- 正在创建 Embedding: {component_key} (Provider: {component_config['provider']}) ---")
        return EmbeddingClass(component_config)
