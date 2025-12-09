from typing import Dict, Any, List, Type
from models.llm_abc import AbstractAgent, AbstractLLM, AbstractTool
from rag.rag_module import RAGModule
from langgraph.graph import StateGraph, END, START 
# from tools_implementations import SearchTool
import asyncio


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

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行混合搜索和 LLM 总结。"""
        query = state.get("input", state.get("query", ""))
            
        print(f"\n[RAG Agent 执行中]：正在对查询 '{query[:20]}...' 执行混合搜索...")
        context_docs = self.rag_module.hybrid_search(query, top_k=2)
        context = "\n".join(context_docs)
        
        # # 使用 LLM 进行总结
        final_answer = await self.llm.generate(f"请基于以下上下文，回答用户的问题：{query}\n上下文:\n{context}")

        # # 模拟 RAG 流程
        # final_answer = (
        #     f"✅ RAG 流程执行成功：根据 RAG 知识库检索结果，查询 '{query[:10]}...' "
        #     f"的答案是：LLM 工厂用于解耦多模型调用，这是 **工厂模式和依赖注入** 架构的核心。"
        # )

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
        
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行数学计算并返回结果。"""
        query = state.get("input", state.get("query", ""))
        
        # ⚠️ 添加执行日志和结果
        print(f"\n[Calculator Agent 执行中]：意图识别为计算，正在执行 Tool 调用...")

        # 模拟 Tool 运行结果
        tool_result = await self.tools['math_solver'].run("12 * 5 + 3") # 假设工具能直接计算表达式

        # 必须返回包含 'output' 键的状态
        return {"output": f"✅ Calculator 流程执行成功：计算查询 '{query[:10]}...' 的结果是：{tool_result}"}
    
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
    def __init__(self, llm: AbstractLLM, tools: Dict[str, AbstractTool], 
                 #rag_module: RAGModule, 
                 config: Dict[str, Any],
                 executor_agents: Dict[str, AbstractAgent]):
        
        # 依赖注入：注入 LLM 实例, Tools 集合, RAG 模块
        self.llm = llm
        self.tools = tools
        # self.rag_module = rag_module
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


    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph 节点的核心处理函数：意图识别和路由决策。"""
        user_input = state.get("input", "")
        print(f"\n[Router Agent 意图识别中]：原始输入：{user_input[:20]}...")
        
        # 意图识别 Prompt
        prompt_llm = (
            f"原始输入：{user_input}，判断用户意图：如果用户在进行数学计算（例如：多少，等于，加，乘），返回 'CALCULATOR'。如果用户在询问知识或概念（例如：是什么，为什么，介绍），返回 'RAG'。否则返回 'DEFAULT'。请只返回一个单词作为结果。"
        )
        prompt_web_search = user_input
        # 第一次调用 LLM 并获取原始响应
        # 创建协程对象 (注意：这里不加 await，只是创建任务)
        llm_task = self.llm.generate(prompt_llm)
        search_tool_task = self.tools['web_search'].run(prompt_web_search)
        # 结果将按任务在 gather 中的顺序返回
        decision_raw, search_result = await asyncio.gather(
            llm_task, 
            search_tool_task
        )

        print(f"  [Router Agent LLM 原生响应]: {decision_raw}") # 打印 LLM 的原生响应
        # 打印搜索结果（用于演示并发已完成）
        print(f"  [Router Agent 并发 Web 搜索结果]: {search_result[:30]}...")

        # 规范化 decision：转换为大写并去除空格
        decision = decision_raw.strip().upper()
    
        # # 模拟调用 LLM
        # decision = self.llm.generate(prompt).strip().upper()


        # 确保 decision 是预期的路由键
        if 'CALCULATOR' in decision:
            decision = "CALCULATOR"
            
        elif 'RAG' in decision:
            decision = "RAG"
        else:
            decision = "DEFAULT"
        
        print(f"  [Router Agent 意图]: 识别为 {decision}")

        # 确保返回一个字典，这是 LangGraph 的要求
        return {"decision": decision,"tools": search_result}
    
    
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
