from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

#示例文本
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]

docs = [Document(doc) for doc in texts] #转换为Document对象,用于嵌入

#嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

#向量数据库
local_chroma_path = "./chroma_db" #本地向量数据库路径
vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory=local_chroma_path
) #chroma向量数据库

print(f"Chroma index has been saved to {local_chroma_path}")

#3.加载索引并执行查询
loaded_vectorstore = Chroma(
    persist_directory=local_chroma_path,
     embedding_function=embeddings
)

query = "FAISS是做什么的?"
results = loaded_vectorstore.similarity_search(query, k=1)

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for doc in results :
    print(f"- {doc.page_content}")