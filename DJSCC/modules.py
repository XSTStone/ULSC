import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # Output: 16x128x128
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output: 32x64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: 64x32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: 128x16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Output: 256x8x8
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: 128x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: 64x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: 32x64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: 16x128x128
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: 3x256x256
            nn.Sigmoid() # Use sigmoid to output values between 0 and 1
        )

    def forward(self, x):
        x = x.view(-1, 256, 8, 8)
        x = self.conv_layers(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    # def forward(self, x):
    #     y = self.encoder(x)
    #     x_hat = self.decoder(y)
    #     return x_hat

# 块丢失信道
def block_dropout(tensor, num_blocks, dropout_p=0.01):
    # 确定每个块的大小
    block_size = tensor.shape[1] // num_blocks
    # 对于不能整除的情况，将多出来的部分加到最后一个块中
    blocks_sizes = [block_size] * (num_blocks - 1) + [tensor.shape[1] - block_size * (num_blocks - 1)]
    # 初始化一个mask
    mask = torch.ones_like(tensor)
    # 生成mask，决定哪些块被置0
    for i in range(num_blocks):
        # 按照dropout_p的概率决定是否置0
        if torch.rand(1).item() < dropout_p:
            start_idx = sum(blocks_sizes[:i])
            end_idx = start_idx + blocks_sizes[i]
            mask[:, start_idx:end_idx] = 0
    # 应用mask置0
    tensor_masked = tensor * mask

    return tensor_masked




