import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=4, stride=2, padding=1),  # 减少通道数
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 减少通道数
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=2, dilation=2),  # 减少通道数
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 减少通道数
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = torch.mean(x, dim=[2, 3])  # 将输出展平
        return self.sigmoid(x)  # 返回经过 Sigmoid 的输出

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             # 第1层：普通卷积，输入通道为4，输出通道为64，卷积核大小为4，步长为2，填充为1
#             nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             # 第2层：膨胀卷积，输入通道为64，输出通道为128，卷积核大小为4，膨胀率为2，填充为2
#             nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=2, dilation=2),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             # 第3层：普通卷积，输入通道为128，输出通道为1，卷积核大小为4，步长为2，填充为1
#             nn.Conv2d(128, 1, kernel_size=4, stride=2, padding=1)
#         )
        
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.model(x)
#         x = torch.mean(x, dim=[2, 3])  # 将输出展平
#         return self.sigmoid(x)  # 返回经过 Sigmoid 的输出

# 测试模型
if __name__ == "__main__":
    model = Discriminator()
    input_tensor = torch.randn(8, 5, 256, 256)  # 示例输入，batch size为8
    output = model(input_tensor)
    print(output.shape)  # 应该输出: torch.Size([8, 1])
