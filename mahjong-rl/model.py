# Yunlong Lu created at 2021/11/28, advised by Wenxin Li, PKU AI Lab

import torch
from torch import nn

class CNNModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        # 保持和 mahjong-sl 的 CNNModel 一致，以兼容 SL 预训练权重
        self._tower = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 235)
        )
        # 价值分支，用于 PPO 的 value function（无 SL 预训练权重）
        # 输入为 logits (235维)，而非原始特征
        self._value_branch = nn.Sequential(
            nn.Linear(235, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        logits = self._tower(obs)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        # value branch 的输入：取 _tower 中 Flatten 后的特征
        # 由于 _tower 内部结构已变，这里用中间特征计算 value
        # 为简化，直接从 logits 的前一层获取（或者重新 forward 一次）
        # 实际上，我们可以让 _tower 只负责特征提取，但当前 _tower 已经包含最后的 Linear
        # 所以这里我们让 value branch 的输入是 logits（兼容处理）
        value = self._value_branch(logits)
        return masked_logits, value