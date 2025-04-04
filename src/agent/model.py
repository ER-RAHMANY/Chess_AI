import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(12, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        # )
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4096),  # 64 from * 64 to = 4096
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ELU(),
            nn.Linear(256, 1),
            nn.Tanh(),  # Output between -1 and 1
        )

    def forward(self, x):
        x = self.conv(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
    
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class AlphaGoNet(nn.Module):
    def __init__(self, input_shape, output_shape, n_res_blocks=5, filters=128):
        super().__init__()

        c, h, w = input_shape
        policy_dim, value_dim = output_shape

        self.initial_conv = nn.Sequential(
            nn.Conv2d(c, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(filters) for _ in range(n_res_blocks)
        ])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * h * w, policy_dim),
            # nn.Sigmoid(),  # If you use BCE loss. For CE, remove this.
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(h * w, 256),
            nn.ReLU(),
            nn.Linear(256, value_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value