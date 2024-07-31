# 导入所需库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import copy

from modules import Autoencoder, block_dropout

# 设置GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    num_epochs = 100 #训练次数
    num_frames = 10 #划分的帧数
    error_value = 0.01 #帧丢失率
    # 设定变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 将图像调整为256x256
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 加载数据集
    val_dataset = datasets.ImageFolder(root='./ILSVRC2010_images_val', transform=transform)
    dataset1,  _ = torch.utils.data.random_split(val_dataset, [20000, len(val_dataset) - 20000])
    val_loader = DataLoader(dataset1, batch_size=32, shuffle=True, num_workers=1)

    # 初始化模型和优化器
    autoencoder = Autoencoder()
    autoencoder.to(device)
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4 )  # 使用Adam优化器

    # 训练循环
    t = tqdm(range(len(range(num_epochs))), desc="epoch")
    for I in t:
        for data in val_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            y = autoencoder.encoder(inputs)
            # z = block_dropout(y, num_frames,error_value)
            x_hat = autoencoder.decoder(y)
            # 计算损失
            loss = criterion(x_hat, inputs)

            # 后向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 打印每个epoch的损失
        print(f'Epoch [{I + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    saved_model = copy.deepcopy(autoencoder.state_dict())
    with open('./model/NO_error_epoch_{}.pth'.format(I), 'wb') as f:
        torch.save({'model': saved_model}, f)

if __name__ == '__main__':
    main()