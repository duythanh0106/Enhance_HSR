"""
Hyperparameter tuning entrypoint using Optuna.

Example:
python3 tune_optuna.py --config universal_best --data_root ./data/Harvard --trials 20 --epochs 120
"""

import argparse
import math
from datetime import datetime

from config import (
    Config,
    ConfigBaseline,
    ConfigProposed,
    ConfigLightweight,
    ConfigSpecTrans,
    ConfigUniversalBest,
)
from train import Trainer

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


def build_config(name: str):
    key = (name or "default").lower()
    if key == "baseline":
        return ConfigBaseline()
    if key == "proposed":
        return ConfigProposed()
    if key == "spectrans":
        return ConfigSpecTrans()
    if key == "lightweight":
        return ConfigLightweight()
    if key == "universal_best":
        return ConfigUniversalBest()
    return Config()


def configure_for_tuning(cfg, args, trial):
    if args.data_root:
        cfg.data_root = args.data_root
        if hasattr(cfg, "apply_dataset_profile"):
            cfg.apply_dataset_profile()

    # Keep architecture fixed, tune only training hyperparameters.
    cfg.loss_type = "combined"
    cfg.optimizer = "adamw"
    cfg.use_augmentation = True
    cfg.use_ema = True
    cfg.best_selection_metric = "composite"
    cfg.save_images = False
    cfg.validate_every = 1
    cfg.save_checkpoint_every = max(args.epochs + 1, 999999)

    # Search space
    cfg.learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    cfg.lr_min = trial.suggest_float("lr_min", 1e-7, 5e-6, log=True)
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    cfg.gradient_clip_norm = trial.suggest_float("gradient_clip_norm", 0.5, 2.0)

    cfg.lambda_l1 = 1.0
    cfg.lambda_sam = trial.suggest_float("lambda_sam", 0.05, 0.25)
    cfg.lambda_ssim = trial.suggest_float("lambda_ssim", 0.1, 0.8)

    cfg.use_two_phase_loss = True
    cfg.loss_phase1_ratio = trial.suggest_float("loss_phase1_ratio", 0.2, 0.6)
    cfg.loss_phase1_sam_scale = trial.suggest_float("loss_phase1_sam_scale", 0.2, 0.7)
    cfg.loss_phase1_ssim_scale = trial.suggest_float("loss_phase1_ssim_scale", 0.1, 0.6)
    cfg.loss_phase_transition_epochs = trial.suggest_int("loss_phase_transition_epochs", 5, 40)

    cfg.warmup_epochs = trial.suggest_int("warmup_epochs", 5, 30)
    cfg.warmup_start_lr = trial.suggest_float("warmup_start_lr", 1e-7, 5e-6, log=True)
    cfg.ema_decay = trial.suggest_float("ema_decay", 0.995, 0.9999)

    # Keep effective training effort reasonable for tuning.
    cfg.num_epochs = args.epochs
    if cfg.train_virtual_samples_per_epoch > 0:
        cfg.train_virtual_samples_per_epoch = min(
            cfg.train_virtual_samples_per_epoch, args.max_virtual_train
        )
    if cfg.val_virtual_samples_per_epoch > 0:
        cfg.val_virtual_samples_per_epoch = min(
            cfg.val_virtual_samples_per_epoch, args.max_virtual_val
        )

    cfg.timestamp = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_optuna_t{trial.number:03d}"
    )
    cfg.refresh_output_paths()
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for HSI-SR training")
    parser.add_argument(
        "--config",
        type=str,
        default="universal_best",
        choices=["default", "baseline", "proposed", "spectrans", "lightweight", "universal_best"],
        help="Base config preset",
    )
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root override")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=120, help="Epochs per trial")
    parser.add_argument("--timeout", type=int, default=0, help="Timeout seconds (0 = no timeout)")
    parser.add_argument("--study_name", type=str, default="hsi_sr_tuning", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Sampler seed")
    parser.add_argument("--max_virtual_train", type=int, default=2500, help="Cap virtual train samples/epoch")
    parser.add_argument("--max_virtual_val", type=int, default=512, help="Cap virtual val samples/epoch")
    args = parser.parse_args()

    if optuna is None:
        raise ImportError("Optuna is not installed. Run: pip install optuna")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=max(5, args.epochs // 8))
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial):
        cfg = build_config(args.config)
        cfg = configure_for_tuning(cfg, args, trial)
        trainer = Trainer(cfg)
        trainer.train()
        score = trainer.best_score
        if not math.isfinite(score):
            raise optuna.TrialPruned("Non-finite best score.")
        return score

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=(None if args.timeout <= 0 else args.timeout),
        show_progress_bar=True,
    )

    best = study.best_trial
    print("\n" + "=" * 70)
    print("Optuna tuning completed")
    print(f"Best trial: {best.number}")
    print(f"Best score: {best.value:.6f}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
