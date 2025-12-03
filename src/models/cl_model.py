from src.models.obs.vggt_obs import VGGTObs
from torch import nn

class CLModel(nn.Module):
    def __init__(self, observer, projector, predictor):
        super().__init__()
        self.observer = observer
        self.projector = projector
        self.predictor = predictor

    def forward(self, image):
        _, _, _, H, W = image.shape
        H = H // self.observer.patch_size
        W = W // self.observer.patch_size
        x, start_idx = self.observer(image)  # 提取特征
        x = x[23]
        x = x[:, :, start_idx:, :]  # 去掉 tokens 部分
        x = self.projector(x)  # 投影特征
        x = self.predictor(x)  # 预测特征
        return x
    



