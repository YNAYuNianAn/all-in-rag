#固定分块
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader

# loader = TextLoader("../../data/C2/txt/蜂医.txt")
# docs = loader.load()

# text_splitter = CharacterTextSplitter(
#     chunk_size = 200,
#     chunk_overlap = 10
# )

# chunks = text_splitter.split_documents(docs) 

# print(f"文本被切分为 {len(chunks)} 个块.\n")
# print("--- 前5个块内容示例 ---")
# for i, chunk in enumerate(chunks[:5]) :
#     print("=" * 60)
#     # chunk 是一个 Document 对象，需要访问它的 .page_content 属性来获取文本
#     print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')


#递归分块
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import TextLoader

# loader = TextLoader("../../data/C2/txt/蜂医.txt")
# docs = loader.load() #document列表

# text_splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n", "。", "，", " ", ""],  #逐渐细粒度
#     chunk_size = 200,
#     chunk_overlap = 10
# )

# chunks = text_splitter.split_documents(docs)
# print(f"文本被切分为 {len(chunks)} 个块.\n")
# print("--- 前5个块内容示例 ---")
# for i, chunk in enumerate(chunks[:5]) :
#     print("=" * 60)
#     # chunk 是一个 Document 对象，需要访问它的 .page_content 属性来获取文本
#     print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')


#语义分块
# from langchain_community.document_loaders import TextLoader
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_community.embeddings import HuggingFaceEmbeddings

# #语义切块使用的嵌入模型
# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-zh-v1.5",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )

# loader = TextLoader("../../data/C2/txt/蜂医.txt")
# docs = loader.load()

# text_splitter = SemanticChunker(
#     embeddings = embeddings,
#     breakpoint_threshold_type="percentile" #百分比
# )

# chunks = text_splitter.split_documents(docs)

# print(f"文本被切分为 {len(chunks)} 个块.\n")
# print("--- 前5个块内容示例 ---")
# for i, chunk in enumerate(chunks[:5]) :
#     print("=" * 60)
#     # chunk 是一个 Document 对象，需要访问它的 .page_content 属性来获取文本
#     print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

loader = TextLoader("../../data/C1/markdown/easy-rl-chapter1.md")
docs = loader.load()
text = docs[0].page_content

# 定义标题层级映射
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
text_spliter = MarkdownHeaderTextSplitter(
    headers_to_split_on = headers_to_split_on,
    strip_headers=False, #是否保留标题文本在内容里
)

chunks = text_spliter.split_text(text) #MarkdownHeaderTextSplitter只接收文本

print(f"文本被切分为 {len(chunks)} 个块.\n")
print("--- 前5个块展示---")
for i, element in enumerate(chunks) :
    print("=" * 60)
    print(f"第 {i+1} 块,(长度: {len(element.page_content)}): {element.page_content}")
