import torch
from diffusers import StableDiffusionPipeline

# 检查是否有可用的 CUDA 设备
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("CUDA 设备不可用，使用 CPU 进行推理。")
    device = torch.device("cpu")

# 从预训练模型加载管道，并指定正确的设备
# pipe = StableDiffusionPipeline.from_pretrained("WYH/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained("WYH/stable-diffusion-v1-4", revision="fp16")
pipe.to(device)

# 提示语句
prompt = "metaverse"

# 在指定设备上进行推理
# 生成图像的尺寸
target_image_size = (256, 256)
image = pipe(prompt, target_image_size=target_image_size).images[0]

# 保存生成的图像
image.save("meta.png")
