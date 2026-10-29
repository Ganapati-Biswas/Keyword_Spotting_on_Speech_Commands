import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        # Input: (B, 1, H, W)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (B, 16, H, W)
            nn.BatchNorm2d(16),                           # (B, 16, H, W)
            nn.ReLU(),                                    # (B, 16, H, W)
            nn.MaxPool2d(2),                              # (B, 16, H/2, W/2)
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B, 32, H/2, W/2)
            nn.BatchNorm2d(32),                            # (B, 32, H/2, W/2)
            nn.ReLU(),                                     # (B, 32, H/2, W/2)
            nn.MaxPool2d(2),                               # (B, 32, H/4, W/4)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, H/4, W/4)
            nn.BatchNorm2d(64),                            # (B, 64, H/4, W/4)
            nn.ReLU(),                                     # (B, 64, H/4, W/4)
            nn.MaxPool2d(2)                                # (B, 64, H/8, W/8)
        )
        self.classifier = nn.Linear(64, num_classes)       # (B, num_classes)

    def forward(self, x):
        x = self.features(x)        # (B, 64, H/8, W/8)
        x = x.mean(dim=[2, 3])      # (B, 64) — Global Avg Pool
        x = self.classifier(x)      # (B, num_classes)
        return x

class DSCNN(nn.Module):
    def __init__(self, num_classes=12, width_mult=1.0):
        super().__init__()
        def ds_conv(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(),
                nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )
        
        c = [int(x * width_mult) for x in [16, 32, 64, 64, 64]]
        self.net = nn.Sequential(
            nn.Conv2d(1, c[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(),
            ds_conv(c[0], c[1], 2),
            ds_conv(c[1], c[2], 2),
            ds_conv(c[2], c[3], 1),
            ds_conv(c[3], c[4], 1)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(c[4], num_classes)

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x

class BCResNet(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        # Placeholder for BCResNet, simple version
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = self.net(x).view(x.size(0), -1)
        return self.classifier(x)

def get_model(name, num_classes=12, **kwargs):
    models = {
        'baselinecnn': BaselineCNN,
        'dscnn': DSCNN,
        'bcresnet': BCResNet
    }
    name = name.lower()
    if name not in models:
        raise ValueError(f"Model {name} not supported.")
    return models[name](num_classes=num_classes, **kwargs)
