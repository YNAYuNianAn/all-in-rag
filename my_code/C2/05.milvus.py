import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image

#常量设置
MODEL_NAME = "BAAI/bge-base-en-v1.5" #嵌入模型
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth" #视觉权重
DATA_DIR = "../../data/C3" #数据
COLLECTION_NAME = "multimodal_demo" #集合名
MILVUS_URI = "http://localhost:19530" #资源请求路径,寻找Milvus服务

#工具设置

#编码器
class Encoder :
    """编码器类, 用于将图像和文本编码为向量"""
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name #所选模型
        self.model_path = model_path #视觉权重
        self.model = Visualized_BGE(
            model_name_bge = model_name,
            model_weight = model_path
        )
        self.eval() #评估模式
    
    def encode_query(self, image_path: str, text: str) -> list[float] :
        """将图片和文本转化为向量"""
        with torch.no_grad() :
            query_emb = self.model.encode(image=image_path, text=text) 

            #Milvus 的 data 参数要求传入 Python 列表 或 NumPy 数组，不接受 PyTorch 张量。
            return query_emb.tolist()[0]
    
    def encode_image(self, image_path: str) -> list[float] :
        with torch.no_grad() :
            query_emb = self.model.encode(image=image_path) 

            #Milvus 的 data 参数要求传入 Python 列表 或 NumPy 数组，不接受 PyTorch 张量。
            return query_emb.tolist()[0]
    


# 3. 初始化客户端
print("--> 正在初始化编码器和Milvus客户端...")
encoder = Encoder(model_name=MODEL_NAME,model_path=MODEL_PATH) #编码器
milvus_client = MilvusClient(uri=MILVUS_URI) #milvus客户端



##开始构建向量数据库(服务器)
# 4. 创建 Milvus Collection
print(f"\n--> 正在创建 Collection '{COLLECTION_NAME}'")
if milvus_client.has_collection(COLLECTION_NAME) :
    milvus_client.drop_collection(COLLECTION_NAME) #删掉原先的集合
    print(f"已删除已存在的 Collection: '{COLLECTION_NAME}'")

image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))

if not image_list :
    raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 图像。")
dim = len(encoder.encode_image(image_list[0])) #生成的向量维度

#字段规范列表
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512)
]

#创建集合 Schema
Schema = CollectionSchema(fields, description="多模态图文检索")
print("Schema 结构:")
print(schema)

#创建集合
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功创建 Collection: '{COLLECTION_NAME}'")
print("Collection 结构:")
print(milvus_client.describe_collection(collection_name=COLLECTION_NAME))