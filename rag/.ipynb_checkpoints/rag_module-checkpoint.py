from typing import List, Dict, Any
from models.llm_abc import AbstractEmbedding 
import chromadb
# --- 导入 LangChain 相关组件 ---
from langchain_community.vectorstores import Chroma
# 修正：EnsembleRetriever 已移至 langchain.retrievers
from langchain.retrievers import EnsembleRetriever 
from langchain.retrievers import BM25Retriever
from langchain_core.documents import Document

# 全局变量用于存储内存中的 Chroma 客户端
_CHROMA_CLIENT = None

class RAGModule:
    """
    RAG 核心模块：负责索引创建、数据摄取和混合搜索逻辑。
    它依赖于一个 AbstractEmbedding 实例进行向量化。
    """
    def __init__(self, embedding_model: AbstractEmbedding, config: Dict[str, Any]):
        """
        初始化 RAG 模块，注入 Embedding 模型。
        """
        self.embedder = embedding_model
        self.config = config
        self.collection_name = config.get("collection_name", "rag_collection")
        self.search_k = config.get("search_k", 5) # 检索文档数量

        # 1. 初始化 Chroma 客户端 (内存模式，或持久化模式)
        global _CHROMA_CLIENT
        if _CHROMA_CLIENT is None:
            # 实际项目中，这里可以配置为 chromadb.PersistentClient 或远程客户端
            _CHROMA_CLIENT = chromadb.Client() 
        self.client = _CHROMA_CLIENT

        self.vectorstore: Chroma | None = None
        self.retriever: EnsembleRetriever | None = None

        print(f"  [RAG] RAGModule 已初始化，使用 Embedding 模型: {self.embedder.__class__.__name__}")
        print(f"  [RAG] ChromaDB 客户端已创建 (内存模式)。集合名称: {self.collection_name}")


    def ingest_data(self, documents: List[str]):
        """
        数据摄取和索引创建过程。
        将文本数据转换为 LangChain Document 结构，并导入 ChromaDB。
        """
        if self.vectorstore is not None:
             # 如果已经初始化过，为了演示效果，可以先清空集合
             self.client.delete_collection(name=self.collection_name)
             self.vectorstore = None
             
        print(f"\n  [RAG] 开始摄取 {len(documents)} 份文档，并创建索引...")
        
        # 1. 将字符串列表转换为 LangChain Document 列表
        lc_documents = [Document(page_content=doc) for doc in documents]

        # 2. **密集索引 (Dense Index):** 使用注入的 AbstractEmbedding 实例创建 Chroma
        
        # *** 兼容处理：为了让 Chroma 正常工作，EmbeddingFactory 需返回一个 LangChain 兼容的实例 ***
        # 我们假设 self.embedder 是 LangChain Embeddings 的一个兼容子集。
        
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedder # <--- 假定它兼容 LangChain Embeddings 接口
        )
        self.vectorstore.add_documents(lc_documents)
        print(f"  [RAG] ✅ 密集向量索引已创建 ({len(lc_documents)} 个文档)。")

        # 3. **稀疏索引 (Sparse Index):** 使用 BM25 创建稀疏索引
        bm25_retriever = BM25Retriever.from_documents(lc_documents)
        bm25_retriever.k = self.search_k 
        print("  [RAG] ✅ 稀疏 BM25 索引已创建。")

        # 4. **组合检索器 (Hybrid Search):** 使用 EnsembleRetriever 合并检索器
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, self.vectorstore.as_retriever(k=self.search_k)],
            weights=[0.5, 0.5], # 稀疏搜索和密集搜索各占 50% 权重
            c=100 # RRF 算法的常数因子
        )
        print("  [RAG] ✅ 混合搜索 (EnsembleRetriever/RRF) 组装完成。")


    def hybrid_search(self, query: str, top_k: int = 5) -> List[str]:
        """
        实现混合搜索逻辑 (Dense + Sparse)，使用 EnsembleRetriever。
        """
        if self.retriever is None:
            # 尝试在搜索前先进行数据摄取（如果可能，但为了演示清晰，最好在 main.py 中明确调用）
            # raise RuntimeError("RAG 模块未进行数据摄取/初始化，请先调用 ingest_data()。")
            pass # 允许继续，但如果未初始化 retriever 会报错

        print(f"\n  [RAG] 正在执行混合搜索 (查询: '{query}') ...")
        
        # 调用 EnsembleRetriever 进行混合搜索和 RRF 重排序
        retrieved_docs: List[Document] = self.retriever.get_relevant_documents(query)

        results = [
            f"--- 检索结果 {i+1} ---\n{doc.page_content}" 
            for i, doc in enumerate(retrieved_docs[:top_k])
        ]
        
        print(f"  [RAG] ✅ 混合搜索完成，返回 {len(retrieved_docs[:top_k])} 个结果。")
        return results