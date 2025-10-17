from typing import Dict, Any, Type, List
# 导入所有工厂
from factory.llm_factory import BaseFactory, LLMFactory
from factory.tools_factory import ToolsFactory
from factory.rag_factory import RAGFactory # 导入 RAGFactory
# 导入抽象接口和所有具体 Agent 实现
from models.llm_abc import AbstractAgent
from models.agents_implementations import RouterAgent, RAGAgent, CalculatorAgent # 导入所有 Agent 类

# --- Agent 注册表 ---
AGENT_MAP: Dict[str, Type[AbstractAgent]] = {
    "router": RouterAgent,         # 意图识别和路由 Agent
    "rag": RAGAgent,               # 新增 RAG 流程 Agent
    "calculator": CalculatorAgent, # 新增 Calculator Tool Agent
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

    # 递归调用辅助函数
    def _get_executor_agents_instances(self, executor_keys: List[str]) -> Dict[str, AbstractAgent]:
        """递归获取所有子执行 Agent 实例。"""
        executor_agents_instances = {}
        for exec_key in executor_keys:
            # 这里的递归调用是 Agent 嵌套的关键
            executor_agents_instances[exec_key] = self.get_instance(exec_key)
        return executor_agents_instances

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
        dependencies = agent_component_config.get("dependencies", {})
        llm_dependency_key = dependencies.get("llm_key")
        tools_dependency_keys = dependencies.get("tools_keys", [])
        rag_dependency_key = dependencies.get("rag_key")
        # 🆕 新增：提取子 Agent 依赖的 key
        executor_keys = dependencies.get("executor_agents", {}) 

        # # 3. 通过注入的工厂获取依赖实例 (LLM, Tools, RAG)
        # llm_instance = self.llm_factory.get_instance(llm_dependency_key) if llm_dependency_key else None

        # 3. 通过注入的工厂获取依赖实例，并存入字典
        agent_dependencies = {} # 用于存储最终传递给 Agent 构造函数的所有依赖

        if llm_dependency_key:
            # 依赖注入 LLM
            agent_dependencies['llm'] = self.llm_factory.get_instance(llm_dependency_key)

        # 依赖注入 Tools (字典)
        tools_instances = {}
        for tool_key in tools_dependency_keys:
            tools_instances[tool_key] = self.tools_factory.get_instance(tool_key)

       # 依赖注入 RAG Module
        if rag_dependency_key:
            agent_dependencies['rag_module'] = self.rag_factory.get_instance(rag_dependency_key)
        agent_dependencies['tools'] = tools_instances
            
        # 依赖注入子 Agent 实例 (执行器)
        if executor_keys:
             agent_dependencies['executor_agents'] = self._get_executor_agents_instances(executor_keys)

        # 4. 始终注入 Agent 自身的配置
        agent_dependencies['config'] = agent_component_config

        # 5. 实例化 Agent 对象，使用字典展开 (解决了 TypeError)
        try:
            agent_instance = AgentClass(**agent_dependencies)
            return agent_instance
        except TypeError as e:
            # 捕获并提供更详细的错误信息
            required_args = AgentClass.__init__.__code__.co_varnames[1:AgentClass.__init__.__code__.co_argcount]
            passed_args = agent_dependencies.keys()
            print(f"❌ 实例化 Agent '{component_key}' 失败，AgentClass: {AgentClass.__name__}")
            print(f"   Agent 构造函数签名 (期望参数): {required_args}")
            print(f"   Factory 实际传递参数: {list(passed_args)}")
            print(f"   原始错误: {e}")
            raise