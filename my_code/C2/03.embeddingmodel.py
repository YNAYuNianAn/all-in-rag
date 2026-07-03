import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

model = Visualized_BGE(
    model_name_bge = "BAAI/bge-base-en-v1.5", #文本嵌入模型底座
    model_weight="../../models/bge/Visualized_base_en_v1.5.pth" #视觉权重
)

model.eval() #评估模式

with torch.no_grad():
    tex_emb = model.encode(text="blue whale") #仅文本嵌入
    img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png") #仅图片嵌入
    #图片和文本混合嵌入
    multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="blue whale")
    img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")
    multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="blue whale")

# 计算相似度
sim_1 = img_emb_1 @ img_emb_2.T
sim_2 = img_emb_1 @ multi_emb_1.T
sim_3 = tex_emb @ multi_emb_1.T
sim_4 = multi_emb_1 @ multi_emb_2.T

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")