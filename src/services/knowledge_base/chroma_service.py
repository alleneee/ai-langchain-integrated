# src/services/knowledge_base/chroma_service.py
import os
import logging
from typing import List, Optional

import chromadb
from langchain_community.document_loaders import (
    DirectoryLoader, UnstructuredFileLoader, PyPDFLoader, Docx2txtLoader, TextLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # 默认使用 OpenAI 嵌入
from langchain_chroma import Chroma
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 配置 ---
# 从环境变量读取配置
DEFAULT_CHROMA_PATH = os.getenv("VECTOR_STORE_DIR", "./my_knowledge_db")  # 从环境变量读取，默认为 ./my_knowledge_db
DEFAULT_COLLECTION_NAME = os.getenv("DEFAULT_COLLECTION_NAME", "my_docs_collection")  # 默认集合名称
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))  # 默认块大小
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))  # 默认块重叠

# 确保 OPENAI_API_KEY 已在你的环境变量或 .env 文件中设置

class ChromaKnowledgeBase:
    """
    管理 ChromaDB 向量存储中文档的加载、处理和存储。
    """

    def __init__(self,
                 persist_directory: str = DEFAULT_CHROMA_PATH,
                 embedding_function = None, # 允许自定义嵌入函数
                 ):
        """
        初始化 ChromaKnowledgeBase。

        Args:
            persist_directory (str): 存储 ChromaDB 数据库的路径。
            embedding_function: Langchain 嵌入函数。默认为 OpenAIEmbeddings。
        """
        self.persist_directory = persist_directory
        
        # 如果使用默认嵌入，确保 OpenAI API 密钥可用
        if embedding_function is None:
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning("未设置 OPENAI_API_KEY 环境变量。使用 OpenAIEmbeddings 需要它。")
                # 你可能想在这里引发错误或以其他方式处理
            self.embedding_function = OpenAIEmbeddings()
        else:
             self.embedding_function = embedding_function

        # 初始化 ChromaDB 客户端
        # 使用 PersistentClient 将数据保存到磁盘
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        logger.info(f"ChromaDB 客户端已初始化。数据将持久化到: {self.persist_directory}")

        # 初始化 LangChain Chroma 接口 (需要时延迟加载)
        self._vector_store_cache = {} # 缓存已初始化的向量存储（按集合）

    def _get_vector_store(self, collection_name: str) -> Chroma:
        """获取或创建 LangChain Chroma 向量存储实例。"""
        if collection_name not in self._vector_store_cache:
            self._vector_store_cache[collection_name] = Chroma(
                client=self._client,
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory # 确保一致性
            )
            logger.info(f"已为集合 '{collection_name}' 初始化 Chroma 向量存储")
        return self._vector_store_cache[collection_name]

    def _load_documents(self, source_path: str) -> List[Document]:
        """从文件或目录加载文档。"""
        documents = []
        if os.path.isdir(source_path):
            # 简单的目录加载器，考虑常见的基于文本的文件
            # 你可能需要根据文件类型使用更具体的加载器
            # 使用 glob 指定文件类型可能更健壮
            try:
                # 示例：仅加载 .txt, .md, .pdf, .docx
                # 注意：DirectoryLoader 可能难以处理混合的复杂类型，如 PDF/DOCX
                # 对于更复杂的场景，遍历文件并使用特定的加载器：
                # for filename in os.listdir(source_path):
                #     filepath = os.path.join(source_path, filename)
                #     if filename.endswith(".pdf"):
                #         loader = PyPDFLoader(filepath)
                #     elif filename.endswith(".docx"):
                #         loader = Docx2txtLoader(filepath)
                #     elif filename.endswith(".txt"):
                #          loader = TextLoader(filepath, encoding='utf-8') # 指定编码
                #     else: # 回退或跳过
                #          continue
                #     documents.extend(loader.load())
                loader = DirectoryLoader(source_path, glob="**/*[.txt|.md|.pdf|.docx]", recursive=True, show_progress=True, use_multithreading=True, loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True})
                documents = loader.load()
                logger.info(f"从目录 {source_path} 加载了 {len(documents)} 个文档")
            except Exception as e:
                 logger.error(f"从目录 {source_path} 加载文档时出错: {e}", exc_info=True)

        elif os.path.isfile(source_path):
            try:
                # 根据文件扩展名确定加载器
                _, ext = os.path.splitext(source_path)
                ext = ext.lower()
                if ext == '.pdf':
                    loader = PyPDFLoader(source_path)
                elif ext == '.docx':
                    loader = Docx2txtLoader(source_path)
                elif ext in ['.txt', '.md', '.py', '.js', '.html', '.css']: # 如果需要，添加更多文本类型
                    loader = TextLoader(source_path, autodetect_encoding=True)
                else:
                     # 对其他类型回退到 UnstructuredFileLoader
                    loader = UnstructuredFileLoader(source_path, mode="elements") # 'elements' 模式通常效果更好

                documents = loader.load()
                logger.info(f"从文件 {source_path} 加载了 {len(documents)} 个部分")
            except Exception as e:
                 logger.error(f"从文件 {source_path} 加载文档时出错: {e}", exc_info=True)
        else:
            logger.warning(f"源路径未找到或无效: {source_path}")
        return documents

    def _split_documents(self, documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """将文档分割成更小的块。"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"将 {len(documents)} 个文档分割成 {len(split_docs)} 个块。")
        return split_docs

    def add_documents(self,
                      source_path: str,
                      collection_name: str = DEFAULT_COLLECTION_NAME,
                      chunk_size: int = DEFAULT_CHUNK_SIZE,
                      chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                      metadata_list: Optional[List[dict]] = None):
        """
        从源路径加载、分割、嵌入和存储文档。

        Args:
            source_path (str): 包含文档的文件或目录的路径。
            collection_name (str): 要使用的 ChromaDB 集合的名称。
            chunk_size (int): 用于分割的文本块大小。
            chunk_overlap (int): 文本块之间的重叠。
            metadata_list (Optional[List[dict]]): 可选的元数据字典列表，每个文档块一个。
                           如果提供，长度必须与块数匹配。
        """
        logger.info(f"开始为源 {source_path} 添加文档过程")

        # 1. 加载文档
        documents = self._load_documents(source_path)
        if not documents:
            logger.warning("未加载任何文档。中止添加过程。")
            return

        # 2. 分割文档
        split_docs = self._split_documents(documents, chunk_size, chunk_overlap)
        if not split_docs:
             logger.warning("分割后未生成任何块。中止添加过程。")
             return
             
        # 3. 如果需要，准备元数据 (可选，基本示例)
        # 你可能想根据来源等生成更复杂的元数据
        if metadata_list and len(metadata_list) != len(split_docs):
             logger.warning(f"元数据列表长度 ({len(metadata_list)}) 与块数 ({len(split_docs)}) 不匹配。忽略提供的元数据。")
             metadatas = None
        elif metadata_list:
            metadatas = metadata_list
        else:
            # 如果未提供，则添加基本的源元数据
            metadatas = [{"source": doc.metadata.get('source', source_path)} for doc in split_docs]
        
        # 在添加之前为文档分配元数据
        for i, doc in enumerate(split_docs):
            if metadatas and i < len(metadatas):
                 # 使用 update 保留加载器中现有的元数据，如文件路径
                 existing_meta = doc.metadata.copy()
                 existing_meta.update(metadatas[i])
                 doc.metadata = existing_meta

        # 4. 获取向量存储并添加文档
        try:
            vector_store = self._get_vector_store(collection_name)
            
            # 提取文本和元数据用于添加
            texts = [doc.page_content for doc in split_docs]
            final_metadatas = [doc.metadata for doc in split_docs]
            
            # 使用 LangChain 的 add_documents 包装器通常更简单
            vector_store.add_documents(documents=split_docs)
            # 或者，直接使用 Chroma 的 add_texts：
            # vector_store.add_texts(texts=texts, metadatas=final_metadatas)

            logger.info(f"已成功向集合 '{collection_name}' 添加 {len(split_docs)} 个块。")

        except Exception as e:
            logger.error(f"向 Chroma 集合 '{collection_name}' 添加文档时出错: {e}", exc_info=True)


    def query(self,
              query_text: str,
              collection_name: str = DEFAULT_COLLECTION_NAME,
              n_results: int = 4,
              filter_metadata: Optional[dict] = None
              ) -> List[Document]:
        """
        在指定的集合中执行相似性搜索。

        Args:
            query_text (str): 要搜索的文本。
            collection_name (str): 要在其中搜索的集合。
            n_results (int): 要返回的相似文档的数量。
            filter_metadata (Optional[dict]): 用于根据元数据过滤结果的可选字典。
                             示例: {"source": "specific_file.pdf"}

        Returns:
            一个按相似度排序的 LangChain Document 对象列表。
        """
        try:
            vector_store = self._get_vector_store(collection_name)
            
            # 使用带相关性分数的相似性搜索以获取分数
            results_with_scores = vector_store.similarity_search_with_relevance_scores(
                query=query_text,
                k=n_results,
                filter=filter_metadata
            )
            
            # 如果需要，可以基于相关性分数阈值过滤结果 (可选)
            # threshold = 0.7 # 示例阈值
            # relevant_docs = [doc for doc, score in results_with_scores if score >= threshold]
            
            relevant_docs = [doc for doc, score in results_with_scores] # 或者只返回所有 top k
            
            logger.info(f"在集合 '{collection_name}' 中为查询找到 {len(relevant_docs)} 个相关文档。")
            return relevant_docs
        except Exception as e:
            logger.error(f"查询 Chroma 集合 '{collection_name}' 时出错: {e}", exc_info=True)
            return []

# --- 示例用法 ---
if __name__ == '__main__':
    # 确保在环境中设置了 OPENAI_API_KEY！
    if not os.getenv("OPENAI_API_KEY"):
        print("错误: 未设置 OPENAI_API_KEY 环境变量。")
        # 提供退出或使用不同嵌入模型的方法
        # 目前，如果找不到密钥，我们将退出
        # exit()
        # 或者，在没有 OpenAI 的情况下使用虚拟/替代嵌入进行测试：
        print("警告: 未设置 OPENAI_API_KEY。查询可能会失败或使用虚拟嵌入。")
        # from langchain.embeddings import FakeEmbeddings
        # kb = ChromaKnowledgeBase(persist_directory="./my_knowledge_db", embedding_function=FakeEmbeddings(size=768))
        kb = ChromaKnowledgeBase(persist_directory=DEFAULT_CHROMA_PATH) # 将使用 OpenAI，如果密钥缺失，稍后可能会出错
    else:
        kb = ChromaKnowledgeBase(persist_directory=DEFAULT_CHROMA_PATH) # 使用默认 OpenAI

    # 定义测试文档的路径
    docs_path = "./docs_to_load"
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        # 创建虚拟文件用于测试
        try:
            with open(os.path.join(docs_path, "file1.txt"), "w", encoding='utf-8') as f:
                f.write("这是第一个文件的内容，关于苹果及其营养价值。")
            with open(os.path.join(docs_path, "file2.txt"), "w", encoding='utf-8') as f:
                f.write("这第二个文件讨论了橙子，一种富含维生素 C 的柑橘类水果。")
            print(f"已在 {docs_path} 创建虚拟目录和文件")
        except IOError as e:
            print(f"创建虚拟文件时出错: {e}")
            exit()

    print(f"\n从 {docs_path} 添加文档")
    kb.add_documents(source_path=docs_path, collection_name=DEFAULT_COLLECTION_NAME)

    # 添加单个文档
    single_doc_path = "./single_doc.txt"
    try:
        with open(single_doc_path, "w", encoding='utf-8') as f:
            f.write("香蕉是一种以钾含量闻名的流行热带水果。")
        print(f"\n添加单个文档: {single_doc_path}")
        kb.add_documents(source_path=single_doc_path, collection_name=DEFAULT_COLLECTION_NAME)
    except IOError as e:
         print(f"创建单个文档文件时出错: {e}")


    # 查询知识库
    query = "告诉我关于富含维生素 C 的水果"
    print(f"\n使用查询 '{query}' 查询集合 '{DEFAULT_COLLECTION_NAME}'")
    results = kb.query(query_text=query, collection_name=DEFAULT_COLLECTION_NAME, n_results=2)

    if results:
        print("\n查询结果:")
        for i, doc in enumerate(results):
            print(f"\n--- 结果 {i+1} ---")
            # 限制输出长度以便阅读
            content_snippet = doc.page_content[:250] + ('...' if len(doc.page_content) > 250 else '')
            print(f"内容: {content_snippet}")
            print(f"元数据: {doc.metadata}")
    else:
        print("未找到结果。")
        
    # 带过滤器的查询示例
    # 根据加载器存储源路径的方式构造预期的源路径（通常是绝对路径）
    try:
        abs_file1_path = os.path.abspath(os.path.join(docs_path, 'file1.txt'))
        query_filtered = "关于苹果的信息"
        print(f"\n使用过滤器 (source='{abs_file1_path}') 查询: '{query_filtered}'")

        results_filtered = kb.query(
            query_text=query_filtered, 
            collection_name=DEFAULT_COLLECTION_NAME, 
            n_results=1,
            # 重要：检查加载器实际存储的元数据。
            # 它可能是绝对路径、相对路径或仅文件名。
            # 使用正确的键（'source' 很常见）和值格式。
            filter_metadata={"source": "docs_to_load/file1.txt"} 
        )
        
        if results_filtered:
            print("\n过滤查询结果:")
            for doc in results_filtered:
                content_snippet = doc.page_content[:250] + ('...' if len(doc.page_content) > 250 else '')
                print(f"内容: {content_snippet}")
                print(f"元数据: {doc.metadata}")
        else:
            print("未找到过滤查询的结果。")
    except Exception as e:
        print(f"过滤查询期间发生错误: {e}")

    print("\n脚本执行完毕。")
# --- 文件结束 ---
