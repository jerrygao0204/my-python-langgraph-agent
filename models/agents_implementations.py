from typing import Dict, Any, List
from models.llm_abc import AbstractAgent, AbstractLLM, AbstractTool
from rag.rag_module import RAGModule
from langgraph.graph import StateGraph, END, START 



# --- Agent 实现：RAGAgent (负责执行 RAG 流程) ---
class RAGAgent(AbstractAgent):
    """专门执行 RAG 流程的 Agent。"""
    def __init__(self, llm: AbstractLLM, tools: Dict[str, AbstractTool], rag_module: RAGModule | None,config: Dict[str, Any]):
        # 依赖注入：注入 LLM 实例和 Tools 实例
        self.llm = llm
        self.rag_module = rag_module
        self.tools = tools
        self.config = config
        self.name = config.get("name", "RAGAgent")
        print(f"  [Agent] RAGAgent '{self.name}' 已初始化。")
        print(f"  [Agent] 依赖 LLM: {self.llm.__class__.__name__}")
        print(f"  [Agent] 依赖 RAG Module: {self.rag_module.__class__.__name__}")
        print(f"  [Agent] 依赖 Tools: {list(tools.keys())}")

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行混合搜索和 LLM 总结。"""
        query = state.get("query", "")
        if not query:
            return {"output": "RAG Agent 失败：查询缺失。", "decision": "END"}
            
        print(f"\n[RAG Agent] -> 执行混合搜索：{query[:20]}...")
        context_docs = self.rag_module.hybrid_search(query, top_k=2)
        context = "\n".join(context_docs)
        
        # 使用 LLM 进行总结
        final_answer = self.llm.generate(f"请基于以下上下文，回答用户的问题：{query}\n上下文:\n{context}")

        return {
            "output": f"[RAG 流程完成]\n[检索上下文]：{context}\n[LLM 最终回复]：{final_answer}", 
            "decision": "END" # RAGAgent 流程结束
        }

    def get_agent_flow(self) -> Any:
        # 子 Agent 通常被视为父图的一个节点，流程由父图定义，但我们仍然需要实现抽象方法
        return self.process # 简单返回 process 方法


# --- Agent 实现：CalculatorAgent (负责执行 Tool 流程) ---
class CalculatorAgent(AbstractAgent):
    """专门执行数学计算和结果总结的 Agent。"""

    def __init__(self, llm: AbstractLLM, tools: Dict[str, AbstractTool], config: Dict[str, Any]):
        self.llm = llm
        # 为了执行 Tool，我们需要工具的引用
        self.tools = tools 
        self.name = config.get("name", "CalculatorAgent")
        print(f"  [Agent] CalculatorAgent '{self.name}' 已初始化。")
        
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """解析表达式，调用 Tool，并进行 LLM 总结。"""
        user_input = state.get("input", "")
        tool_name = "math_solver" # 硬编码工具键名
        
        # 【核心逻辑：将表达式解析和 Tool 调用移到这里，实现完全解耦】
        try:
            # 1. 表达式解析（沿用 RouterAgent 中修复后的逻辑）
            calc_text = user_input.split("计算", 1)[-1].strip()
            if '等于' in calc_text:
                calculation_text = calc_text.split("等于", 1)[0].strip()
            else:
                calculation_text = calc_text.strip()
            
            expression = (
                calculation_text
                .replace("乘以", "*")
                .replace("加", "+")
                .replace("减", "-")
                .replace("除以", "/")
                .replace("除", "/")
                .replace(" ", "")
            )

            # 2. 实际运行 Tool
            tool_output = self.tools[tool_name].run(expression) 
            print(f"\n[Calculator Agent] -> 工具结果：{tool_output[:20]}...")
            
            # 3. 使用 LLM 格式化回复
            final_answer = self.llm.generate(f"用户问题：{user_input}。工具结果：{tool_output}。请基于工具结果简洁回复。")

            return {
                "output": f"[Tool 流程完成]\n[LLM 最终回复]：{final_answer}", 
                "decision": "END"
            }

        except Exception as e:
            return {
                "output": f"[Tool 流程出错]: 无法计算或解析表达式 '{user_input}'. 错误: {e}", 
                "decision": "END"
            }

    def get_agent_flow(self) -> Any:
        # 简单返回 process 方法
        return self.process

# --- Agent 实现：RouterAgent (更新构造函数和 get_agent_flow) ---
class RouterAgent(AbstractAgent):
    """
    Agent 路由器：模拟根据用户输入进行意图识别和流程选择。
    现在它注入了子 Agent 实例，并将执行逻辑委托给它们。
    """
    # def __init__(self, llm: AbstractLLM, tools: Dict[str, AbstractTool], rag_module: RAGModule, 
    #              config: Dict[str, Any], **executor_agents: AbstractAgent): # ⬅️ 注入子 Agent
    def __init__(self, llm: AbstractLLM, tools: Dict[str, AbstractTool], rag_module: RAGModule, config: Dict[str, Any], executor_agents: Dict[str, AbstractAgent]):
        
        # 依赖注入：注入 LLM 实例, Tools 集合, RAG 模块
        self.llm = llm
        self.tools = tools
        self.rag_module = rag_module
        self.config = config
        self.name = config.get("name", "DefaultRouter")
        self.executor_agents = executor_agents

        # 强制检查关键子 Agent 是否存在 (根据 config.yaml 中的 key)
        required_keys = ["rag_executor", "calc_executor"]
        missing_executors = [k for k in required_keys if k not in self.executor_agents]
        
        # 注入子 Agent 实例
        # 必须使用与 AgentFactory 注入时相同的 key
        self.rag_executor = executor_agents.get("rag_executor")     
        self.calc_executor = executor_agents.get("calc_executor")   
        
        # 确保子 Agent 已注入
        if not self.rag_executor or not self.calc_executor:
             raise RuntimeError("RouterAgent 启动失败：缺少必要的执行 Agent (rag_executor 或 calc_executor)。")

        if missing_executors:
            # 修正：捕获运行时错误（这是上一个问题中解决的逻辑）
            raise RuntimeError(f"RouterAgent 启动失败：缺少必要的执行 Agent: {missing_executors}。")

        print(f"  [Agent] RouterAgent '{self.name}' 已初始化。")
        print(f"  [Agent] 依赖 LLM: {self.llm.__class__.__name__}")
        print(f"  [Agent] 依赖 Tools: {list(tools.keys())}")
        print(f"  [Agent] 委托执行 Agents: [RAG: {self.rag_executor.name}, CALC: {self.calc_executor.name}]")


    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph 节点的核心处理函数：意图识别和路由决策。"""
        user_input = state.get("input", "")
        print(f"\n[Router Agent 意图识别中]：原始输入：{user_input[:20]}...")
        
        # 意图识别 Prompt
        prompt = (
            f"原始输入：{user_input}\n"
            "判断用户意图：如果用户在进行数学计算（例如：多少，等于，加，乘），返回 'CALCULATOR'。\n"
            "如果用户在询问知识或概念（例如：是什么，为什么，介绍），返回 'RAG'。\n"
            "否则返回 'DEFAULT'。\n"
            "请只返回一个单词作为结果。"
        )

        # 模拟调用 LLM
        decision = self.llm.generate(prompt).strip().upper()

        # 确保 decision 是预期的路由键
        if 'CALCULATOR' in decision:
            decision = "CALCULATOR"
        elif 'RAG' in decision:
            decision = "RAG"
        else:
            decision = "DEFAULT"
        
        print(f"  [Router Agent 意图]: 识别为 {decision}")

        # 确保返回一个字典，这是 LangGraph 的要求
        return {"decision": decision}
    
    
    def get_agent_flow(self) -> Any:
        """
        定义 LangGraph 流程图 (Flow)。
        LangGraph 节点现在委托给子 Agent 的 process 方法。
        """
        workflow = StateGraph(Dict[str, Any])

        # 1. 添加节点 (节点现在是子 Agent 的 process 方法)
        # 注意：节点名 (CALCULATOR, RAG) 必须与 RouterAgent.process 的返回值对应
        workflow.add_node("route", self.process)
        workflow.add_node("CALCULATOR", self.calc_executor.process) 
        workflow.add_node("RAG", self.rag_executor.process)         

        # 2. 设置起点
        workflow.set_entry_point("route")

        # 3. 添加条件边 (Conditional Edges)
        # 路由节点 (route) 根据其返回的 'decision' 键进行跳转
        workflow.add_conditional_edges(
            "route",
            lambda x: x["decision"],
            {
                "CALCULATOR": "CALCULATOR",  # 跳转到 CalculatorAgent 节点
                "RAG": "RAG",                # 跳转到 RAGAgent 节点
                "DEFAULT": END,              # 默认情况直接结束
            },
        )
        
        # 4. 添加普通边 (Normal Edges)
        # 两个执行 Agent 完成后，流程直接结束
        workflow.add_edge("CALCULATOR", END)
        workflow.add_edge("RAG", END)

        print(f"[✔ 架构框架搭建完成]：RouterAgent 已重构，LangGraph 流程已使用子 Agent 接入。")
        return workflow.compile()
