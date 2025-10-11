from typing import Dict, Any, Type
from factory.llm_factory import BaseFactory
from factory.embedding_factory import EmbeddingFactory # 导入 EmbeddingFactory
from rag.rag_module import RAGModule

# --- RAG 注册表 ---
RAG_MAP: Dict[str, Type[RAGModule]] = {
    "chroma": RAGModule, # 暂时使用 RAGModule 封装所有功能
}

class RAGFactory(BaseFactory):
    """
    RAG Factory：负责创建 RAG 流程模块。
    它依赖于 EmbeddingFactory 来获取向量化模型。
    """
    def __init__(self, full_config: Dict[str, Any], embed_factory: EmbeddingFactory):
        super().__init__(full_config)
        # 依赖注入：注入 EmbeddingFactory
        self.embed_factory = embed_factory

    def get_instance(self, component_key: str) -> RAGModule:
        """
        component_key: 配置中 rag 部分的键名，如 'primary_vector_store'。
        """
        config_key = "rag"
        
        # 1. 获取 RAG 自身的配置和类
        component_config, RAGClass = self._get_config_and_class(config_key, component_key, RAG_MAP)
        
        print(f"\n--- 正在组装 RAG 模块: {component_key} (Type: {RAGClass.__name__}) ---")

        # 2. 从配置中提取依赖组件的 key
        dependencies = component_config.get("dependencies", {})
        embed_dependency_key = dependencies.get("embed_key")

        # 3. 通过注入的工厂获取依赖实例
        embedding_instance = None
        if embed_dependency_key and isinstance(embed_dependency_key, str):
            # 依赖注入 Embedding 实例
            embedding_instance = self.embed_factory.get_instance(embed_dependency_key)
        
        if not embedding_instance:
             raise RuntimeError(f"RAG 模块 '{component_key}' 必须配置一个有效的 Embedding 模型依赖。")
             
        # 4. 实例化 RAGModule，并将依赖注入
        rag_module_instance = RAGClass(
            embedding_model=embedding_instance,
            config=component_config
        )
        return rag_module_instance
