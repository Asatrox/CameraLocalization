import torch
from src.training.trainer import train
from src.models.cl_model import CLModel
from src.models.obs.vggt_obs import VGGTObs
from src.models.proj.direct_projector import DirectBEVProjector
from src.models.pred.mlp_predictor import MLPPredictor
from src.data.gibson_dataloader import build_gibson_dataloader
from src.training.loss_fn import F3locLoss
from src.models.proj.CA_projector import CABEVProjector

if __name__ == "__main__":
    observer = VGGTObs()
    projector = DirectBEVProjector()
    predictor = MLPPredictor()
    model = CLModel(observer=observer, projector=projector, predictor=predictor)

    # 冻结VGGT模型的参数
    for param in model.observer.parameters():
        param.requires_grad = False     
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    # 创建dataloader
    train_loader, eval_loader, test_loader = build_gibson_dataloader(
        dataset_dir="./data",
        L = 4,
        depth_suffix="depth40",
        batch_size=16,
        num_workers=4,
    )

    # 创建损失函数
    loss_fn = F3locLoss(shape_loss_weight=20)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(
        model = model,
        train_loader = train_loader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        epoches = 20,
        device = device,
        log_dir = "./logs/direct_proj_experiment",
    )
    
    