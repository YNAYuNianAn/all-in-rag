import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader #解析文本
from langchain_text_splitters import RecursiveCharacterTextSplitter #切块
from langchain_huggingface import HuggingFaceEmbeddings #嵌入模型
from langchain_core.vectorstores import InMemoryVectorStore #向量库
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

#加载环境变量
load_dotenv()

#markdown文件路径
markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

#准备数据/加载文本
loader = UnstructuredMarkdownLoader(markdown_path) #加载器
docs = loader.load() #文件夹

#文本分块
text_splitter = RecursiveCharacterTextSplitter()
#文本分块器
chunks = text_splitter.split_documents(docs) 


#中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device':"cpu"},
    encode_kwargs={'normalize_embeddings': True}
)

#向量存储库
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

#提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:""")

llm = ChatDeepSeek(
    model="deepseek-v4-pro",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


#用户查询
question = "文中举了哪些例子"

#检索文档
retrieved_docs = vectorstore.similarity_search(question, k = 3)
docs_content = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs 
)

#生成
answer = llm.invoke(prompt.format(question=question,context=docs_content))
content = answer.content
print(content)