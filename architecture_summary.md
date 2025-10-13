AI 系统架构总结与上下文恢复 (RAG 重构阶段)
当前阶段： 已完成第一阶段“架构框架搭建”。正准备进入第二阶段“RAG 模块重构”（实现混合搜索）。

1. 整体架构图 (工厂模式与依赖注入)
我们的系统核心是一个三级工厂结构，通过依赖注入 (DI) 实现高度解耦：

一级工厂 (原子组件)： ToolsFactory, LLMFactory, EmbeddingFactory

二级工厂 (模块组件)： RAGFactory

三级工厂 (流程组装)： AgentFactory (LangGraph 入口)

graph TD
    subgraph Config
        C[config.yaml] -->|驱动| A;
    end


    subgraph Factories
        A[LLMFactory] -->|注入| D;
        B[EmbeddingFactory] -->|注入| D;
        B -->|注入| E;
        D[ToolsFactory] -->|注入| F;
        E[RAGFactory] -->|注入| F;
        F[AgentFactory]
    end
    
    subgraph Models_And_Tools
        G(AbstractLLM)
        H(AbstractTool)
        I(AbstractEmbedding)
        J(AbstractAgent)
        K(RAGModule)
    end
    
    F -->|组装| J;
    A -->|创建| G;
    D -->|创建| H;
    B -->|创建| I;
    E -->|创建| K;

| 文件/目录                          | 核心职责                                                                        | 依赖关系                   |
|-----------------------------------|--------------------------------------------------------------------------------|---------------------------|
| config/config.yaml                | 配置驱动源：定义所有 LLM, Tools, Agents, RAG 的具体参数和依赖关系。                 | 无                        |
| config/config.py                  | 加载模块：负责读取 YAML 文件，提供配置字典。                                       | 依赖 pyyaml                |
| models/llm_abc.py                 | 抽象层：定义 AbstractLLM, AbstractEmbedding, AbstractTool, AbstractAgent 接口。  | 无                         |
| models/implementations.py         | 原子实现：包含 GPTModel, HuggingFaceModel 等具体实现类。                          | 依赖 llm_abc               |
| models/tools_implementations.py   | 工具实现：包含 CalculatorTool, SearchTool 等具体工具实现类。                      | 依赖 llm_abc               |
| models/agents_implementations.py  | Agent 实现：包含 RouterAgent 等流程决策者。                                      | 依赖 llm_abc               |
| rag/rag_module.py                 | RAG 核心：包含 RAGModule，负责数据摄取和混合搜索逻辑（待实装）。                    | 依赖 llm_abc               |
| factory/llm_factory.py            | LLM 工厂：创建 LLM 实例。包含 BaseFactory (通用工厂基类)。                        | 依赖 models/implementations.py |
| factory/embedding_factory.py      | Embedding 工厂：创建 Embedding 实例。继承 BaseFactory。                          | 依赖 llm_factory           |
| factory/tools_factory.py          | Tools 工厂：创建 Tool 实例。继承 BaseFactory。                                   | 依赖 llm_factory           |
| factory/rag_factory.py            | RAG 工厂：创建 RAGModule 实例。依赖 EmbeddingFactory。                           | 依赖 embedding_factory     |
| factory/agent_factory.py          | Agent 工厂：创建 Agent 流程。依赖 LLMFactory 和 ToolsFactory。                   | 依赖所有底层工厂           |
| main.py                           | 程序入口：加载配置，初始化所有工厂，演示 Agent/RAG 流程调用。                       | 依赖所有工厂               |



AI 系统架构总结与上下文恢复 (RAG 重构阶段)
当前阶段： 已完成第一阶段**“架构框架搭建”和“RAG 混合搜索实现”**。系统已成功通过 LangGraph 流程演示了意图路由、工具调用和 RAG 查询，证明了工厂模式和依赖注入的有效性。

下一步： 准备进入**“LangGraph 子流程接入”**，将当前的单个 RouterAgent 解耦为多个独立 Agent（如 CalculatorAgent, RAGAgent），以实现更复杂的子图嵌套。

1. 整体架构图：工厂模式与依赖注入 (DI)
系统核心采用三级工厂结构和依赖注入实现高度解耦。

graph TD
    subgraph Config
        C[config.yaml] -->|驱动| A;
    end


    subgraph Factories
        A[LLMFactory] -->|注入| D;
        B[EmbeddingFactory] -->|注入| D;
        B -->|注入| E;
        D[ToolsFactory] -->|注入| F;
        E[RAGFactory] -->|注入| F;
        F[AgentFactory]
    end
    
    subgraph Models_And_Tools
        G(AbstractLLM)
        H(AbstractTool)
        I(AbstractEmbedding)
        J(AbstractAgent)
        K(RAGModule)
    end
    
    F -->|组装| J;
    A -->|创建| G;
    D -->|创建| H;
    B -->|创建| I;
    E -->|创建| K;

2. 核心文件与职责总结
文件/目录

核心职责

依赖关系

config/config.yaml

配置驱动源：定义所有 LLM, Tools, Agents, RAG 的具体参数和依赖关系。

无

models/llm_abc.py

抽象层： 定义 AbstractLLM, AbstractEmbedding, AbstractTool, AbstractAgent 接口。

无

models/implementations.py

LLM/Embedding 实现： 包含 GPTModel, OpenAIEmbeddingsModel 等具体实现类。

依赖 llm_abc

models/tools_implementations.py

工具实现： 包含 CalculatorTool 等具体工具实现类。

依赖 llm_abc

rag/rag_module.py

RAG 核心： 包含 RAGModule，实现了 混合搜索（EnsembleRetriever/RRF） 和数据摄取。

依赖 llm_abc

factory/*.py

工厂层： 实现 BaseFactory 及其继承类，负责组件的实例化和依赖注入。

交叉依赖

main.py

程序入口： 加载配置，初始化所有工厂，演示 LangGraph 流程调用（意图路由 -> CALCULATOR/RAG）。

依赖所有工厂

3. LangGraph 流程及状态修正总结
当前 LangGraph 流程由 RouterAgent.get_agent_flow() 定义，负责顶层路由：

START -> route (RouterAgent.process)

route (基于 state['input'] 识别意图，返回 state['decision'] 键)

decision: "CALCULATOR" -> CALCULATOR 节点

decision: "RAG" -> RAG 节点

decision: "END" -> END

CALCULATOR/RAG 节点 -> END

关键修正点 (已完成)：

状态类型错误 (TypeError): 已通过确保 RouterAgent.process 始终返回一个字典（包含 decision 键）来解决。

RAG 依赖注入错误 (AttributeError): 已通过在 main.py 中，使用 router_agent.rag_module 同一实例来执行 ingest_data，解决了 RAG 检索器未初始化的 Bug。

4. 下一步任务：Agent 子流程解耦
为了实现更复杂的流程控制，我们将按照学习计划，把当前的执行逻辑从 RouterAgent 中剥离出来：

创建 RAGAgent 类：该 Agent 封装 RAG 流程的执行逻辑 (execute_rag 的内容)。

创建 CalculatorAgent 类：该 Agent 封装 Tool 流程的执行逻辑 (execute_tool 的内容)。

更新 AgentFactory：注册新的 rag 和 calculator Agent 类型。

重构 RouterAgent：它的 get_agent_flow 将不再直接调用执行函数，而是将路由目标指向新的 子 Agent 实例。




