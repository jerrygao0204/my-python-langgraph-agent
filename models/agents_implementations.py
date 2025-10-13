from typing import Dict, Any, List, Type
from models.llm_abc import AbstractAgent, AbstractLLM, AbstractTool
from rag.rag_module import RAGModule
from langgraph.graph import StateGraph, END, START 
from functools import partial

# --- LangGraph 流程节点函数 ---
def execute_rag(state: Dict[str, Any], rag_module: RAGModule, llm: AbstractLLM) -> Dict[str, Any]:
    """LangGraph 节点：执行 RAG 流程。"""
    query = state.get("query", "")
    if not query or not rag_module:
        return {"output": "RAG 流程失败：查询或模块缺失。", "decision": "END"}
        
    print(f"\n[LangGraph RAG Node] -> 执行混合搜索：{query[:20]}...")
    context_docs = rag_module.hybrid_search(query, top_k=2)
    context = "\n".join(context_docs)
    
    # 使用 LLM 进行总结
    final_answer = llm.generate(f"请基于以下上下文，回答用户的问题：{query}\n上下文:\n{context}")

    return {"output": f"[RAG 流程完成]\n[检索上下文]：{context}\n[最终回复]：{final_answer}", "decision": "END"}


def execute_tool(state: Dict[str, Any], llm: AbstractLLM) -> Dict[str, Any]:
    """LangGraph 节点：执行 Tool 流程并总结。"""
    tool_output = state.get("tool_output", "N/A")
    query = state.get("query", "N/A")

    print(f"\n[LangGraph Tool Node] -> 工具结果：{tool_output[:20]}...")

    # 使用 LLM 格式化回复
    final_answer = llm.generate(f"用户问题：{query}。工具结果：{tool_output}。请基于工具结果简洁回复。")
    
    return {"output": f"[Tool 流程完成]\n[LLM 最终回复]：{final_answer}", "decision": "END"}
        
# --- Agent 实现：RouterAgent 模拟 ---
class RouterAgent(AbstractAgent):
    """
    Agent 路由器：模拟根据用户输入进行意图识别和流程选择。
    这个 Agent 依赖于一个 LLM 和一个 Tools 集合。
    """
    def __init__(self, llm: AbstractLLM, tools: Dict[str, AbstractTool], rag_module: RAGModule | None,config: Dict[str, Any]):
        # 依赖注入：注入 LLM 实例和 Tools 实例
        self.llm = llm
        self.tools = tools
        self.rag_module = rag_module
        self.config = config
        self.name = config.get("name", "DefaultRouter")
        
        print(f"  [Agent] RouterAgent '{self.name}' 已初始化。")
        print(f"  [Agent] 依赖 LLM: {self.llm.__class__.__name__}")
        print(f"  [Agent] 依赖 Tools: {list(tools.keys())}")
        print(f"  [Agent] 依赖 RAG 模块: {self.rag_module.__class__.__name__ if self.rag_module else 'NoneType'}") # 【更新打印信息】

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """模拟意图识别和路由决策。"""
        user_input = state.get("input", "")
        print(f"  [Router Agent 意图识别中]：原始输入：{user_input}")
        # 模拟调用 LLM 进行意图识别
        # decision = self.llm.generate(f"判断用户意图：{user_input}。返回'RAG'或'SEARCH'。")

        # 假设 RAG 是关于“架构”或“流程”的，而 SEARCH 是关于“计算”的
        if "计算" in user_input or "多少" in user_input:
            decision = "CALCULATOR"
        elif "RAG" in user_input or "架构" in user_input or "LLM工厂" in user_input or "LangGraph" in user_input:
             decision = "RAG"
        else:
            decision = "SEARCH"
            
        print(f"  [Router Agent 意图]: 识别为 {decision}")

         # 准备一个临时的状态更新字典
        state_update: Dict[str, Any] = {"decision": decision, "query": user_input}

        # --- LangGraph 路由修改点 ---
        if decision == "CALCULATOR":
            # 准备 Tool 流程的输入和调用
            tool_name = "math_solver"
            # 修正表达式提取 Bug
            try:
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

                tool_output = self.tools[tool_name].run(expression) 
                
                # 添加 tool_output 到状态更新中
                state_update["tool_output"] = tool_output

            except Exception as e:
                # 如果工具调用或解析失败，路由到 END
                print(f"Tool 流程出错: {e}")
                state_update["output"] = f"[Tool 流程出错]: {e}"
                state_update["decision"] = "END" # 强制路由到 END

        elif decision == "RAG":
            # RAG 流程只需要 query 和 decision，不需要额外更新
            pass
        
        # 返回最终的状态更新字典。LangGraph 会将此字典合并到全局状态中
        return state_update




    def get_agent_flow(self) -> Any:
            """
            返回 Agent 流程的 LangGraph 结构或可执行入口。
            AgentFactory 将调用此方法来获取最终可运行的流程。
            """
            # 1. 定义图结构和状态
            workflow = StateGraph(Dict[str, Any])
            
            # 2. 添加节点
            # Router Agent 作为第一个节点，调用其自身的 process 方法
            workflow.add_node("route", self.process)
            
            # 定义 RAG 节点和 Tool 节点 (使用 partial 注入依赖)
            rag_node = lambda state: execute_rag(state, self.rag_module, self.llm)
            tool_node = lambda state: execute_tool(state, self.llm)
            
            workflow.add_node("RAG", rag_node)
            workflow.add_node("CALCULATOR", tool_node)
    
            # 3. 设置边和条件路由
            workflow.add_edge(START, "route")

            # **【核心修改点：定义路由函数，从状态字典中提取 decision 键】**
            def route_decision(state: Dict[str, Any]) -> str:
                """根据 'route' 节点返回的状态中的 'decision' 键进行路由。"""
                return state.get("decision", "END")
            
            workflow.add_conditional_edges(
                "route", 
                # 将路由函数传递给 LangGraph
                route_decision, 
                {
                    "RAG": "RAG",
                    "CALCULATOR": "CALCULATOR",
                    "END": END,
                }
            )
            
            # 4. 设置结束边
            workflow.add_edge("RAG", END)
            workflow.add_edge("CALCULATOR", END)
            
            # 编译流程图
            return workflow.compile()

        # if decision == "CALCULATOR":
        #     # 路由到 TOOL 流程 (CALCULATOR)
        #     tool_name = "math_solver"
        #     # 仅提取表达式，不包含问题句式
        #     expression = user_input.split("乘以", 1) 
        #     expression = expression[-1].replace("等于多少？", "").strip() if len(expression) > 1 else "12 * 5"
            
        #     tool_output = self.tools[tool_name].run(expression)
            
        #     # 使用 LLM 格式化回复
        #     llm_response = self.llm.generate(f"用户问题：{user_input}。工具结果：{tool_output}。请基于工具结果简洁回复。")
        #     return {"output": f"[Router Agent 决定]：路由到 TOOL 流程。\n[Tool 结果]：{tool_output}\n[LLM 回复]：{llm_response}"}

        # elif decision == "RAG":
        #     # 路由到 RAG 流程
        #     if not self.rag_module:
        #          return {"output": "[Router Agent 决定]：路由到 RAG 流程，但 RAG 模块未注入。"}
                 
        #     # 【实际调用 RAG 模块进行检索】
        #     context_docs = self.rag_module.hybrid_search(user_input, top_k=2)
        #     context = "\n".join(context_docs)
            
        #     # 模拟用 LLM 进行总结 
        #     llm_response = self.llm.generate(f"请基于以下上下文，回答用户的问题：{user_input}\n上下文:\n{context}")

        #     return {"output": f"[Router Agent 决定]：路由到 RAG 流程。\n[RAG 检索上下文]：{context}\n[LLM 总结回复]：{llm_response}"}
            
            
        
        # # 模拟执行工具
        # if "SEARCH" in decision:
        #     search_result = self.tools['web_search'].run(user_input)
        #     return {"output": f"[Router Agent 决定]：调用搜索工具。结果: {search_result}"}
        # else:
        #     return {"output": f"[Router Agent 决定]：路由到 RAG 流程。"}
            

# --- 其他 Agent 实现（例如 RAGAgent, PlannerAgent 等将添加到这里） ---
# ...
