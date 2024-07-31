import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from modules import Autoencoder, block_dropout
from similarity import runs
from psnr import runpsnr
import pandas as pd

# 设置GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化函数，读取数据集目录和预处理的配置。
        :param root_dir: 数据集目录的路径。
        :param transform: torchvision.transforms中的数据预处理。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        """
        返回数据集中图像的数量。
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        根据索引idx返回其中一个数据点和其标签。
        """
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image



def ftest(autoencoder, data_loader, num_frames, p):
    t = 0
    autoencoder.eval()
    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)
            t = t + 1
            save_image(images, os.path.join('resizedata', 'image{}.png'.format(t)))
            # y = autoencoder.encoder(images)
            # z = block_dropout(y, num_frames, p)
            # x_hat = autoencoder.decoder(z)
            # # x_hat = autoencoder.decoder(y)
            # save_image(x_hat, os.path.join('outputdata', 'image{}.png'.format(t)))


# 这里可以添加模型训练的代码

def main():
    # P = [0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1]
    P = [0.001]
    num_frames=14

    model = Autoencoder().to(device)
    model.load_state_dict(torch.load('./model/NO_error_epoch_99.pth')['model'])

    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 缩放图像为256x256
        transforms.ToTensor(),  # 将图像转换为Tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    # 创建CustomDataset实例
    # dataset = CustomDataset(root_dir='./testdata', transform=transform)
    root_dir = './testdata'
    file_list = sorted(os.listdir(root_dir))

    # 手动创建排序后的文件路径列表
    file_paths = [os.path.join(root_dir, filename) for filename in file_list]
    # 创建CustomDataset实例
    dataset = CustomDataset(file_paths, transform=transform)
    # 使用DataLoader来加载数据集
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for p in P:
        print(f'帧丢失率: {p}')
        # SM = []
        PSNR = []
        for ii in range(1, 2):
            print(f'****{ii}****')
            ftest(model, data_loader, num_frames, p)
            # sm = runs()
        #     psnr = runpsnr()
        #     # SM.append(sm)
        #     PSNR.append(psnr)
        # # print('sm:', np.mean(SM))
        # print('psnr:', np.mean(PSNR))
        # data = {'帧丢失率': [p] * len(PSNR), 'PSNR': PSNR}
        # df = pd.DataFrame(data)
        # filename = f'NO14psnr_results_{p}.xlsx'
        # df.to_excel(filename, index=False)



if __name__ == '__main__':
    main()