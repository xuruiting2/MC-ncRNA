# 这个文件是让两个模型共享loss的
import torch
import torch.optim as optim

class SharedOptimizer:
    def __init__(self, modelA, learning_rate=0.00001):
        # 创建一个优化器，包含两个模型的参数
        self.optimizer = optim.Adam(modelA.parameters(), lr=learning_rate)

    def update_models(self, shared_loss, modelA):
        # 在每个迭代中更新参数
        self.optimizer.zero_grad()
        self.optimizer.step()
        torch.nn.utils.clip_grad_norm_(modelA.parameters(), 1.0)  # 梯度裁