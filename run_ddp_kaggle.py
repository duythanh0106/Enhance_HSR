import argparse
import os

import torch
import torch.distributed as dist

import train as train_mod
from config import build_config
from train import ModelEMA


def parse_args():
    """Parse CLI arguments cho DDP launcher."""
    parser = argparse.ArgumentParser(description="Kaggle DDP launcher for Enhance_HSR")
    parser.add_argument("--config", type=str, default="chikusei")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--train_virtual", type=int, default=None)
    parser.add_argument("--val_virtual", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lambda_l1", type=float, default=None)
    parser.add_argument("--lambda_sam", type=float, default=None)
    parser.add_argument("--lambda_ssim", type=float, default=None)
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        choices=["l1", "l2", "combined", "adaptive"],
    )
    parser.add_argument("--gradient_clip_norm", type=float, default=0.5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--mixed_precision", action="store_true")
    return parser.parse_args()


def main():
    """Khởi tạo DDP process group, build Trainer và chạy training với DistributedDataParallel."""
    args = parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if rank != 0:
        from tqdm import tqdm as _tqdm

        def _quiet_tqdm(*targs, **tkwargs):
            tkwargs["disable"] = True
            return _tqdm(*targs, **tkwargs)

        train_mod.tqdm = _quiet_tqdm

    cfg = build_config(args.config)
    cfg.data_root = args.data_root
    if hasattr(cfg, "apply_dataset_profile"):
        cfg.apply_dataset_profile()

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.patch_size is not None:
        cfg.patch_size = args.patch_size
    if args.train_virtual is not None:
        cfg.train_virtual_samples_per_epoch = args.train_virtual
    if args.val_virtual is not None:
        cfg.val_virtual_samples_per_epoch = args.val_virtual
    if args.max_epochs is not None:
        cfg.num_epochs = args.max_epochs
    if args.warmup_epochs is not None:
        cfg.warmup_epochs = args.warmup_epochs
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.lambda_l1 is not None:
        cfg.lambda_l1 = args.lambda_l1
    if args.lambda_sam is not None:
        cfg.lambda_sam = args.lambda_sam
    if args.lambda_ssim is not None:
        cfg.lambda_ssim = args.lambda_ssim
    if args.loss_type is not None:
        cfg.loss_type = args.loss_type
    if args.gradient_clip_norm is not None:
        cfg.gradient_clip_norm = args.gradient_clip_norm
    if args.resume:
        cfg.resume = True
        cfg.resume_checkpoint = args.resume

    # Kaggle/DDP runtime defaults.
    cfg.cache_in_memory = True
    cfg.mixed_precision = bool(args.mixed_precision)
    cfg.device = f"cuda:{local_rank}"
    cfg.log_device_runtime = (rank == 0)
    cfg.seed = int(getattr(cfg, "seed", 42)) + rank
    cfg.quiet_tqdm = True
    cfg.refresh_output_paths()

    if rank == 0:
        cfg.print_config()
        print(f"DDP world_size={world_size}, local_rank={local_rank}")

    trainer = train_mod.Trainer(cfg)
    trainer.model = torch.nn.parallel.DistributedDataParallel(
        trainer.model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    # Rebuild optimizer/scheduler after DDP wrapping.
    # Trainer.build_* restores previous states when available.
    trainer.optimizer = trainer.build_optimizer()
    trainer.scheduler = trainer.build_scheduler()
    if cfg.use_ema:
        trainer.ema = ModelEMA(trainer.model, decay=cfg.ema_decay)

    orig_save_checkpoint = trainer.save_checkpoint
    if rank == 0:
        def _save_checkpoint_ddp_compat(epoch, metrics, is_best=False, selection_score=None):
            model_ref = trainer.model
            if hasattr(model_ref, "module"):
                trainer.model = model_ref.module
            try:
                return orig_save_checkpoint(epoch, metrics, is_best, selection_score)
            finally:
                trainer.model = model_ref

        trainer.save_checkpoint = _save_checkpoint_ddp_compat
    else:
        trainer._log = lambda *largs, **lkwargs: None
        trainer.save_checkpoint = lambda *largs, **lkwargs: None

    try:
        trainer.train()
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
