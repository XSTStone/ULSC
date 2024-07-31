import torch
from torchvision.transforms import ToTensor
from PIL import Image
import math
import numpy as np

# 读取图像并转换为Tensor
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return ToTensor()(image)

# 计算PSNR
def calculate_psnr(image1, image2):
    mse = torch.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def runpsnr():
    PSNR = []
    for i in range(1, 101):
        # 示例图片路径
        image_path1 = f'./resizedata/image{i}.png'
        image_path2 = f'./outputdata/image{i}.png'

        # 加载图片
        image1 = load_image(image_path1)
        image2 = load_image(image_path2)

        # 计算PSNR
        psnr_value = calculate_psnr(image1, image2)
        # print(f"PSNR value: {psnr_value} dB")
        PSNR.append(psnr_value)
    return np.mean(PSNR)
