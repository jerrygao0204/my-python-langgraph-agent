from typing import Dict, Any, Type
from models.llm_abc import AbstractTool
from models.tools_implementations import CalculatorTool, SearchTool
from factory.llm_factory import BaseFactory # 从 LLM Factory 导入 BaseFactory

# --- 工具注册表 ---
TOOL_MAP: Dict[str, Type[AbstractTool]] = {
    "calculator": CalculatorTool, # tool_config 中 key 是 type: calculator
    "search": SearchTool,        # tool_config 中 key 是 type: search
}

class ToolsFactory(BaseFactory):
    """Tools Factory，继承自 BaseFactory，负责创建工具实例。"""
    def get_instance(self, component_key: str) -> AbstractTool:
        """
        component_key: 配置中 tools 部分的键名，如 'math_solver' 或 'web_search'
        """
        config_key = "tools"
        component_config, ToolClass = self._get_config_and_class(config_key, component_key, TOOL_MAP)

        # 实例化并返回 Tool 对象
        # 注意: Tool 依赖于 config 中的 'type' 字段查找类，而不是 'provider'
        print(f"\n--- 正在创建 Tool: {component_key} (Type: {component_config['type']}) ---")
        return ToolClass(component_config)
