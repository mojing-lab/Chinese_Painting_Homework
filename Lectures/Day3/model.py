import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # ——— 特征提取部分 ———
        self.feature = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),      # (1) 卷积：输入1通道→6通道, 5×5
            nn.Tanh(),                           # (2) 激活：Tanh
            nn.AvgPool2d(2),                     # (3) 平均池化 2×2, stride=2
            nn.Conv2d(6, 16, kernel_size=5),     # (4) 卷积：6→16, 5×5
            nn.Tanh(),                           # (5) 激活
            nn.AvgPool2d(2),                     # (6) 池化
            nn.Conv2d(16, 120, kernel_size=5),   # (7) 卷积：16→120, 5×5
            nn.Tanh(),                           # (8) 激活
        )
        # ——— 分类部分 ———
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),                 # (9) 全连接：120→84
            nn.Tanh(),                          # (10) 激活
            nn.Linear(84, num_classes)         # (11) 全连接：84→类别数
        )

    def forward(self, x):
        x = self.feature(x)                    # → [batch,120,1,1]
        x = x.flatten(1)                       # → [batch,120]
        return self.classifier(x)              # → [batch,num_classes]


class LeNet5_Reg(nn.Module):
    def __init__(self, num_classes=10, p_dropout=0.5):
        super().__init__()
        # ——— 特征提取：加入 BatchNorm + ReLU ———
        self.feature = nn.Sequential(
            nn.Conv2d(1, 6, 5),            
            nn.BatchNorm2d(6),               # 批归一化
            nn.ReLU(),                       # ReLU 替代 Tanh
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 5),
            nn.ReLU(),
        )
        # ——— 分类：加入 Dropout ———
        self.classifier = nn.Sequential(
            nn.Dropout(p=p_dropout),         # 丢弃率 p_dropout
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)                 # → [batch,120,1,1]
        x = x.flatten(1)                    # → [batch,120]
        return self.classifier(x)


