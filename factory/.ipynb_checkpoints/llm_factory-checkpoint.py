from typing import Dict, Any, Type
from models.llm_abc import AbstractLLM, AbstractEmbedding
# 导入具体实现。注意：需要确保 models.implementations 存在且导入路径正确
from models.implementations import GPTModel, HuggingFacePipelineModel 

class BaseFactory:
    """所有工厂的抽象基类，封装配置注入和通用的实例获取逻辑。"""
    def __init__(self, full_config: Dict[str, Any]):
        # 将完整的配置注入到工厂中
        self.config = full_config
    
    def _get_config_and_class(self, config_key: str, component_key: str, REGISTRY_MAP: Dict[str, Type[Any]]) -> tuple[Dict[str, Any], Type[Any]]:
        """
        通用查找逻辑：根据配置键和组件键获取配置字典和对应的模型类。
        """
        # 1. 安全获取顶层配置块 (e.g., self.config['llm'])
        top_level_config = self.config.get(config_key)
        
        # 【修正点一】：检查 top_level_config 是否为 None 或非字典类型
        if not isinstance(top_level_config, dict):
            # 如果配置块未找到或为空（被解析为 None），则安全地设置为空字典
            top_level_config = {}

        # 2. 从顶层配置块中获取指定组件的配置字典 (e.g., self.config['llm']['prod_model'])
        component_config = top_level_config.get(component_key)
        
        # 【修正点二】：如果组件配置仍未找到或为空
        if not component_config:
            raise ValueError(f"组件配置键 '{component_key}' 未在 '{config_key}' 配置块中找到。请检查 config.yaml。")

        # 3. 从组件配置中获取提供者 (provider) 或类型 (type)
        component_type = component_config.get("provider", component_config.get("type", "")).lower()
        ComponentClass = REGISTRY_MAP.get(component_type)

        if not ComponentClass:
            raise ValueError(f"不支持的组件提供者或类型: {component_type}")
        
        return component_config, ComponentClass


# 注册表：将配置中的 provider 映射到实际的 Python 类
LLM_MAP: Dict[str, Type[AbstractLLM]] = {
    "openai": GPTModel,
    "huggingface": HuggingFacePipelineModel,
}

class LLMFactory(BaseFactory):
    """LLM Factory，继承自 BaseFactory，负责创建 LLM 实例。"""
    def get_instance(self, component_key: str) -> AbstractLLM:
        """
        component_key: 配置中 llm 部分的键名，如 'prod_model'。
        """
        config_key = "llm"
        component_config, LLMClass = self._get_config_and_class(config_key, component_key, LLM_MAP)
        
        # 实例化并返回 LLM 对象，将该组件的配置传入
        print(f"\n--- 正在创建 LLM: {component_key} (Provider: {component_config['provider']}) ---")
        return LLMClass(component_config)
