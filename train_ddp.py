def setup_distributed():
    """从 torchrun 的环境变量中读取 rank / world_size / local_rank。"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

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
    train_loader, eval_loader, test_loader = build_gibson_dataloader_ddp(
        dataset_dir="./data/gibson_f",
        L=4,
        depth_suffix="depth40",
        batch_size=16,
        num_workers=4,
        world_size=world_size,
        rank=rank,
    )

    # 创建损失函数
    loss_fn = F3locLoss(shape_loss_weight=20)

    train_ddp(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epoches=20,
        device=device,
        log_dir="./logs/direct_proj_experiment",
        rank=rank,
    )

    cleanup_distributed()


if __name__ == "__main__":
    main()