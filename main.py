from factory.llm_factory import LLMFactory
from factory.embedding_factory import EmbeddingFactory
from factory.tools_factory import ToolsFactory
from factory.agent_factory import AgentFactory
from factory.rag_factory import RAGFactory
from config.config import load_config
from typing import Dict, Any
from langgraph.graph import StateGraph, END, START 
import copy 
import asyncio

async def run_agent_flow(app_flow: StateGraph, query: str):
    """
    运行 LangGraph 流程。
    """
    # 流程开始时的初始状态
    initial_state = {"input": query, "query": query, "output": "", "decision": ""}
    
    print(f"\n--- LangGraph 流程调用演示 ---")
    
    # 运行流程
    try:
        final_state = await app_flow.ainvoke(initial_state)

        # 打印最终结果
        print("--------------------------------------------------")
        print(f"原始输入: {query}")
        print(f"LangGraph 最终结果:")
        print(final_state.get('output', '[没有最终输出]'))
        print("--------------------------------------------------")
    except Exception as e:
        print(f"❌ LangGraph 流程执行失败: {e}")
        print("--------------------------------------------------")

async def main():
    """主程序入口，加载配置，初始化工厂并获取所需组件。"""
    
    # --- 1. 加载配置 ---
    try:
        full_config = load_config()
    except Exception as e:
        print(f"系统启动失败，请检查配置文件 'config/config.yaml' 和依赖库 'pyyaml'. 错误: {e}")
        return

    print("\n--- 系统启动：初始化工厂 ---")
    
    # 实例化所有底层工厂
    llm_factory = LLMFactory(full_config)
    embed_factory = EmbeddingFactory(full_config)
    tools_factory = ToolsFactory(full_config)
    rag_factory = RAGFactory(full_config, embed_factory) # 注入 EmbeddingFactory
    agent_factory = AgentFactory(full_config, llm_factory, tools_factory, rag_factory)

    # --- 2. 从 Agent Factory 获取核心 Agent 流程 ---

    # 获取主要 Agent 路由器
    router_agent = agent_factory.get_instance("primary_router")

    # 获取编译后的 LangGraph 流程
    app_flow = router_agent.get_agent_flow()


    # --- 3. 从 RAG Factory 获取 RAG 模块并演示 ---
    # rag_processor = router_agent.rag_module
    rag_processor = router_agent.rag_executor.rag_module

    if rag_processor is None:
        raise RuntimeError("RAG 模块未成功注入到 RouterAgent 中。请检查 AgentFactory 和 config.yaml。")


    # 演示 RAG 数据摄取
    documents_to_ingest = [
        "LLM工厂用于解耦多模型调用，是架构的核心。",
        "混合搜索结合了稀疏和密集索引，以提高检索准确性。",
        "什么是提高RAG准确性的方法？和搜索准确性有关吗？",
        "LangGraph是一个强大的工具，用于构建复杂的Agent流程。",

    ]

    # ⚠️ 确保 RAGModule 在使用前被初始化 (数据摄取)
    print("\n--- 正在初始化 RAG 模块数据 ---")
    rag_processor.ingest_data(documents_to_ingest)
    print("--- RAG 模块数据初始化完成 ---\n")

    # --- 4. 运行流程演示 ---
    print("[✔ 架构框架搭建完成]：已进入 LangGraph 流程编排阶段。")
    
    # 案例一：CALCULATOR 流程 (路由 -> CalculatorAgent)
    await run_agent_flow(app_flow, "帮我计算 (12 乘以 5) 加上 3 等于多少？")

    # 案例二：RAG 流程 (路由 -> RAGAgent)
    await run_agent_flow(app_flow, "帮我查询LLM工厂用于解耦多模型调用，这是什么架构的核心？")
    
    # 案例三：DEFAULT 流程 (路由 -> END)
    await run_agent_flow(app_flow, "今天天气真好，我们应该去哪里野餐？")
    

if __name__ == "__main__":
    import os
    if not os.path.exists('config'):
        os.makedirs('config')
    
    # 假设 load_config 函数已定义在 config.config 模块中
    # 运行主函数
    asyncio.run(main())
