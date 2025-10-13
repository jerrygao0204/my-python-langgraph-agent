from typing import Dict, Any, Type
# 导入所有工厂
from factory.llm_factory import BaseFactory, LLMFactory
from factory.tools_factory import ToolsFactory
from factory.rag_factory import RAGFactory
# 导入抽象接口和具体 Agent 实现
from models.llm_abc import AbstractAgent
from models.agents_implementations import RouterAgent

# --- Agent 注册表 ---
AGENT_MAP: Dict[str, Type[AbstractAgent]] = {
    "router": RouterAgent, # 意图识别和路由 Agent
    # "rag": RAGAgent, # 未来添加
}

class AgentFactory(BaseFactory):
    """
    Agent Factory：负责组装 Agent 流程。
    它依赖于 LLMFactory 和 ToolsFactory 来获取组件。
    """
    def __init__(self, full_config: Dict[str, Any], llm_factory: LLMFactory, tools_factory: ToolsFactory, rag_factory: RAGFactory):
        super().__init__(full_config)
        # 依赖注入：将其他工厂注入到 AgentFactory 中
        self.llm_factory = llm_factory
        self.tools_factory = tools_factory
        self.rag_factory = rag_factory

    def get_instance(self, component_key: str) -> AbstractAgent:
        """
        component_key: 配置中 agents 部分的键名，如 'primary_router'。
        """
        config_key = "agents"
        
        # 1. 获取 Agent 自身的配置和类
        agent_config_section = self.config.get(config_key, {})
        if not agent_config_section.get(component_key):
             raise ValueError(f"Agent 配置键 '{component_key}' 未在 '{config_key}' 配置块中找到。")

        agent_component_config = agent_config_section[component_key]
        agent_type = agent_component_config.get("type", "").lower()
        AgentClass = AGENT_MAP.get(agent_type)
        
        if not AgentClass:
            raise ValueError(f"不支持的 Agent 类型: {agent_type}")
        
        print(f"\n--- 正在组装 Agent: {component_key} (Type: {agent_type}) ---")

        # 2. 从配置中提取依赖组件的 key
        llm_dependency_key = agent_component_config.get("dependencies", {}).get("llm_key")
        tools_dependency_keys = agent_component_config.get("dependencies", {}).get("tools_keys", [])
        rag_dependency_key = agent_component_config.get("dependencies", {}).get("rag_key")

        # 3. 通过注入的工厂获取依赖实例
        llm_instance = None
        if llm_dependency_key:
            # 依赖注入 LLM
            llm_instance = self.llm_factory.get_instance(llm_dependency_key)
        
        tools_instances = {}
        for tool_key in tools_dependency_keys:
            # 依赖注入 Tools
            tools_instances[tool_key] = self.tools_factory.get_instance(tool_key)

        rag_instance = None
        if rag_dependency_key:
            # 依赖注入 rag
            rag_instance = self.rag_factory.get_instance(rag_dependency_key)
            # print(f"[Agent] 依赖 RAG 模块: {rag_instance.__class__.__name__}")
            
        # 4. 实例化 Agent，并将所有依赖注入
        agent_instance = AgentClass(
            llm=llm_instance,
            tools=tools_instances,
            config=agent_component_config,
            # rag=rag_instance,
            rag_module=rag_instance
        )

        
        return agent_instance
