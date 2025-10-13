from factory.llm_factory import LLMFactory
from factory.embedding_factory import EmbeddingFactory
from factory.tools_factory import ToolsFactory
from factory.agent_factory import AgentFactory
from factory.rag_factory import RAGFactory # 导入新的 RAGFactory
from config.config import load_config
from typing import Dict, Any

def main():
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
    
    # --- 3. 从 RAG Factory 获取 RAG 模块并演示 ---
    # rag_processor = rag_factory.get_instance("primary_vector_store")
    rag_processor = router_agent.rag_module

    if rag_processor is None:
        raise RuntimeError("RAG 模块未成功注入到 RouterAgent 中。请检查 AgentFactory 和 config.yaml。")
    
    
    # 演示 RAG 数据摄取
    documents_to_ingest = [
        "LLM工厂用于解耦多模型调用，是架构的核心。",
        "混合搜索结合了稀疏和密集索引，以提高检索准确性。",
        "什么是提高RAG准确性的方法？和搜索准确性有关吗？",
        "LangGraph是一个强大的工具，用于构建复杂的Agent流程。",

    ]
    rag_processor.ingest_data(documents_to_ingest)
    
    # # 演示 RAG 混合搜索
    # search_query = "什么是提高RAG准确性的方法？"
    # retrieved_docs = rag_processor.hybrid_search(search_query)
    # print(f"\n[RAG 搜索结果]: {retrieved_docs}")

    # 演示 RAG 混合搜索
    search_query = "什么是提高RAG准确性的方法？和搜索准确性有关吗？"
    print(f"\n--- 正在演示 RAG 混合搜索：查询：{search_query} ---")
    retrieved_docs = rag_processor.hybrid_search(search_query, top_k=3) # 检索前3个
    
    # 打印检索结果
    print("\n[RAG 搜索结果]:")
    for i, doc_content in enumerate(retrieved_docs):
        print(f"  [{i+1}] {doc_content}")
    print("-----------------------------------------------------")
    

    # # 演示 Agent 流程调用 (保持不变)
    # print("\n--- Agent 流程调用演示 ---")
    # user_state = {"input": "帮我计算 12 乘以 5 等于多少？"}
    # final_state = router_agent.process(user_state)
    # print(f"原始输入: {user_state['input']}")
    # print(f"Agent 最终结果: {final_state['output']}")

    # --- 4. 演示 Agent LangGraph 流程调用 ---
    # 获取 Router Agent 流程（这里 AgentFactory 会调用 RouterAgent.get_agent_flow()）
    agent_flow = router_agent.get_agent_flow()

    print("\n--- LangGraph 流程调用演示 (计算) ---")
    
    # 案例 1: 计算问题 (应该路由到 CALCULATOR 节点)
    calc_input = "帮我计算 12 乘以 5 等于多少？"
    
    # 运行 LangGraph 流程
    # 注意：LangGraph 的输入是一个字典，包含 "input" 键
    
    result = agent_flow.invoke({"input": calc_input}) 
    
    print(f"原始输入: {calc_input}")
    print(f"LangGraph 最终结果:\n{result['output']}")
    
    # 案例 2: RAG 问题 (应该路由到 RAG 节点)
    rag_input = "LLM工厂用于解耦多模型调用，这是什么架构的核心？"

    print("\n--- LangGraph 流程调用演示 (RAG 知识查询) ---")
    result = agent_flow.invoke({"input": rag_input})
    
    print(f"原始输入: {rag_input}")
    print(f"LangGraph 最终结果:\n{result['output']}")
    
    print("\n[✔ 架构框架搭建完成]：已进入 LangGraph 流程编排阶段。")
    

if __name__ == '__main__':
    main()
