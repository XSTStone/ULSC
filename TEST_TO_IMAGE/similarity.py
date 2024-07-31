import CLIP
import torch
from PIL import Image
import numpy as np
# 加载预训练的CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = CLIP.load("ViT-B/32", device=device)


# 图片预处理
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)  # 添加batch维度并移至正确的设备
    return image


# 计算两张图片之间的语义相似度
def calculate_semantic_similarity(image_path1, image_path2, model):
    image1 = load_and_preprocess_image(image_path1)
    image2 = load_and_preprocess_image(image_path2)

    # 使用CLIP模型提取特征
    with torch.no_grad():
        image_features1 = model.encode_image(image1)
        image_features2 = model.encode_image(image2)

    # 计算余弦相似度
    similarity = torch.cosine_similarity(image_features1, image_features2)
    return similarity.item()

def runs():
    # 示例
    sm=[]
    for i in range(1, 101):
        image_path1 = f'./resizedata/image_{i}.png'
        image_path2 = f'./outputdata/image_image_{i}.png'
        similarity = calculate_semantic_similarity(image_path1, image_path2, model)
        sm.append(similarity)
        # print(f'{i}图片之间的语义相似度为: {similarity}')
    return np.mean(sm)

