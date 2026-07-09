import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.ett import ETTConfig, load_ett_datasets
# NOTE: Keep optional deps out of import-time (e.g., statsmodels for ARIMA).
from models.autoformer_backbone import AutoformerBackbone
from models.gpt4ts_backbone import GPT4TSBackbone
from models.bayesian_head import BayesianLinearHead
from models.tcn_backbone import TCNBackbone
from ompb.bound import compute_variance_proxy_constants
from utils.config import apply_cli_overrides, load_configs
from utils.logging import make_run_dir, save_config_snapshot
from utils.seed import set_seed


def parse_kv_args(argv) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for arg in argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            parsed[k] = v
    return parsed


def _as_bool(v: Optional[str], default: bool) -> bool:
    if v is None:
        return default
    return v not in ("0", "false", "False", "no", "No")


def _pick_device(requested: str) -> str:
    if isinstance(requested, str) and requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA requested ({requested}) but not available; falling back to cpu")
        return "cpu"
    return requested


def find_latest_run_with_files(base_dir: str, prefix: str, required_files: List[str]) -> str:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"log_dir not found: {base_dir}")
    candidates = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    candidates.sort(reverse=True)
    for d in candidates:
        run_dir = os.path.join(base_dir, d)
        if all(os.path.exists(os.path.join(run_dir, f)) for f in required_files):
            return run_dir
    raise FileNotFoundError(
        f"No run found for prefix={prefix} in {base_dir} containing files: {required_files}"
    )


def build_backbone(model_name: str, cfg_models: Dict, cfg_ompb: Dict, meta: Dict, device: str):
    if model_name == "tcn":
        return TCNBackbone(
            in_channels=int(meta.get("x_channels", len(meta["feature_names"]))),
            pred_len=cfg_ompb["pred_len"],
            widths=cfg_models["tcn"]["widths"],
            kernel_size=cfg_models["tcn"]["kernel_size"],
            dropout=cfg_models["tcn"]["dropout"],
            lr=cfg_models["tcn"]["lr"],
            epochs=cfg_models["tcn"]["epochs"],
            batch_size=cfg_models["tcn"]["batch_size"],
            device=device,
            out_channels=int(meta.get("y_channels", 1)),
        )
    if model_name == "autoformer":
        return AutoformerBackbone(
            n_features=int(meta.get("x_channels", len(meta["feature_names"]))),
            pred_len=cfg_ompb["pred_len"],
            d_model=cfg_models["autoformer"]["d_model"],
            n_heads=cfg_models["autoformer"]["n_heads"],
            e_layers=cfg_models["autoformer"]["e_layers"],
            d_layers=cfg_models["autoformer"].get("d_layers", 1),
            moving_avg=cfg_models["autoformer"]["moving_avg"],
            dropout=cfg_models["autoformer"]["dropout"],
            factor=cfg_models["autoformer"].get("factor", 3.0),
            d_ff=cfg_models["autoformer"].get("d_ff"),
            label_len=cfg_models["autoformer"].get("label_len"),
            lr=cfg_models["autoformer"]["lr"],
            epochs=cfg_models["autoformer"]["epochs"],
            batch_size=cfg_models["autoformer"]["batch_size"],
            device=device,
            out_channels=int(meta.get("y_channels", 1)),
        )
    if model_name == "gpt4ts":
        cfg = cfg_models.get("gpt4ts", {})
        return GPT4TSBackbone(
            seq_len=cfg_ompb["seq_len"],
            pred_len=cfg_ompb["pred_len"],
            out_channels=int(meta.get("y_channels", len(meta["feature_names"]))),
            patch_size=int(cfg.get("patch_size", 16)),
            stride=int(cfg.get("stride", 8)),
            d_model=int(cfg.get("d_model", 768)),
            gpt_layers=int(cfg.get("gpt_layers", 2)),
            lr=float(cfg.get("lr", 1e-4)),
            epochs=int(cfg.get("epochs", 5)),
            batch_size=int(cfg.get("batch_size", 16)),
            device=device,
            pretrained_name_or_path=str(cfg.get("pretrained_name_or_path", "gpt2")),
        )
    raise ValueError(f"Unknown model {model_name}")


def _save_backbone_checkpoint(backbone, model_name: str, run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    if model_name in ["tcn", "autoformer", "gpt4ts"]:
        torch.save(backbone.model.state_dict(), os.path.join(run_dir, "backbone.pt"))
    else:
        joblib.dump(backbone, os.path.join(run_dir, "backbone.joblib"))


def _load_backbone_from_dir(model_name: str, run_dir: str, cfg_models: Dict, cfg_ompb: Dict, meta: Dict, device: str):
    backbone = build_backbone(model_name, cfg_models, cfg_ompb, meta, device)
    state = torch.load(os.path.join(run_dir, "backbone.pt"), map_location=device)
    backbone.model.load_state_dict(state)
    return backbone


class _ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


def _sample_windows(dataset, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(dataset) == 0:
        raise ValueError("Source dataset is empty; cannot sample windows for offline training.")
    n = min(int(n), len(dataset))
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=n, replace=False)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for i in idxs:
        x, y = dataset[int(i)]
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.stack(xs, axis=0).astype(np.float32), np.stack(ys, axis=0).astype(np.float32)


def _train_head(
    head: BayesianLinearHead,
    backbone,
    x_train: np.ndarray,
    y_train: np.ndarray,
    device: str,
    lr: float,
    epochs: int,
    batch_size: int,
    lambda_pb: float,
) -> None:
    ds = _ArrayDataset(x_train, y_train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    optim = torch.optim.Adam(head.parameters(), lr=float(lr))
    head.train()
    for _ in range(int(epochs)):
        for x, y in loader:
            base = backbone.predict_batch(x).to(device)
            y = y.to(device)
            pred = head.forward_mean(base)
            mse = torch.mean((pred - y) ** 2)
            kl = head.kl_to_prior()
            loss = mse + float(lambda_pb) * kl / max(1.0, float(ds.__len__()))
            optim.zero_grad()
            loss.backward()
            optim.step()
    head.eval()


def main() -> None:
    args = parse_kv_args(sys.argv)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_ett_cfg, base_ompb_cfg, models_cfg = load_configs(root_dir, args.get("config_path"))

    # Support sweep mode via pred_lens=24,48,96,... (comma-separated).
    if "pred_lens" in args:
        pred_lens = [int(x) for x in args["pred_lens"].split(",") if x.strip() != ""]
        if not pred_lens:
            raise ValueError("pred_lens was provided but empty. Example: pred_lens=24,48,96")
    else:
        pred_lens = [int(args["pred_len"])] if "pred_len" in args else [int(base_ompb_cfg["pred_len"])]

    model_name = args.get("model", "all")
    reuse_backbone = _as_bool(args.get("reuse_backbone"), True)
    head_epochs = int(args.get("head_epochs", "10"))
    head_batch_size = int(args.get("head_batch_size", "64"))

    for pred_len in pred_lens:
        ett_cfg = copy.deepcopy(base_ett_cfg)
        ompb_cfg = copy.deepcopy(base_ompb_cfg)
        local_args = dict(args)
        local_args["pred_len"] = str(pred_len)
        apply_cli_overrides(local_args, ett_cfg, ompb_cfg)
        set_seed(int(ompb_cfg.get("seed", 0)))

        device = _pick_device(str(ompb_cfg.get("device", "cpu")))

        ett = ETTConfig(
            etth1_path=ett_cfg["data"]["etth1_path"],
            etth2_path=ett_cfg["data"]["etth2_path"],
            split=ett_cfg["data"]["split"],
            seq_len=ett_cfg["window"]["seq_len"],
            pred_len=ett_cfg["window"]["pred_len"],
            scaler_path=ett_cfg["scaling"]["scaler_path"],
            align_columns=str(ett_cfg.get("data", {}).get("align_columns", "strict")),
            impute_nan=bool(ett_cfg.get("scaling", {}).get("impute_nan", False)),
        )
        datasets = load_ett_datasets(ett)
        meta = datasets["meta"]
        train_dataset = datasets["etth1"]["train"]

        models_to_train = [model_name] if model_name != "all" else ["tcn", "autoformer", "gpt4ts"]
        for name in models_to_train:
            # Create the run dir for the full offline pipeline (backbone + PMPB artifacts).
            run_dir = make_run_dir(ompb_cfg["log_dir"], f"train_{name}_sl{ett_cfg['window']['seq_len']}_pl{pred_len}")

            backbone_src_dir: Optional[str] = None
            if reuse_backbone:
                required = ["backbone.joblib"] if name in ["arima", "krr"] else ["backbone.pt"]
                prefix = f"backbone_{name}_sl{ett_cfg['window']['seq_len']}_pl{pred_len}"
                try:
                    backbone_src_dir = find_latest_run_with_files(ompb_cfg["log_dir"], prefix, required)
                except FileNotFoundError:
                    backbone_src_dir = None

            if backbone_src_dir is not None:
                print(f"[offline] Loading backbone from {backbone_src_dir}")
                backbone = _load_backbone_from_dir(name, backbone_src_dir, models_cfg, ompb_cfg, meta, device)
            else:
                print(f"[offline] Training backbone for {name} pred_len={pred_len}")
                backbone = build_backbone(name, models_cfg, ompb_cfg, meta, device)
                backbone.fit(train_dataset)

            _save_backbone_checkpoint(backbone, name, run_dir)

            # Sample source windows for training the Bayesian head and for the online source buffer.
            source_buf_n = int(ompb_cfg.get("source_buf_N", 256))
            head_train_n = int(args.get("head_train_N", str(max(source_buf_n, 1024))))
            x_train, y_train = _sample_windows(train_dataset, n=head_train_n, seed=int(ompb_cfg.get("seed", 0)))

            # Fit Bayesian head ("PMPB" offline stage)
            head = BayesianLinearHead(
                pred_len=int(ompb_cfg["pred_len"]),
                out_channels=int(meta.get("y_channels", 1)),
                sigma0=float(ompb_cfg["sigma0"]),
                alpha_init=float(ompb_cfg.get("alpha_init", -6.0)),
                alpha_prior=float(ompb_cfg.get("alpha_prior", -6.0)),
            ).to(device)
            _train_head(
                head=head,
                backbone=backbone,
                x_train=x_train,
                y_train=y_train,
                device=device,
                lr=float(ompb_cfg["train_head_lr"]),
                epochs=head_epochs,
                batch_size=head_batch_size,
                lambda_pb=float(ompb_cfg.get("lambda_pb", 1.0)),
            )
            torch.save(head.state_dict(), os.path.join(run_dir, "head.pt"))

            # Save source buffer used during online calibration
            # (subsample from the same pool for determinism + speed).
            rng = np.random.default_rng(int(ompb_cfg.get("seed", 0)))
            buf_n = min(source_buf_n, x_train.shape[0])
            buf_idxs = rng.choice(x_train.shape[0], size=buf_n, replace=False)
            source_x = x_train[buf_idxs].astype(np.float32)
            source_y = y_train[buf_idxs].astype(np.float32)
            np.save(os.path.join(run_dir, "source_x.npy"), source_x)
            np.save(os.path.join(run_dir, "source_y.npy"), source_y)

            # Compute variance-proxy constants used in the PAC-Bayes bound term.
            constants = compute_variance_proxy_constants(
                source_dataset=_ArrayDataset(source_x, source_y),
                backbone=backbone,
                head=head,
                alpha=float(ompb_cfg["alpha"]),
                beta=float(ompb_cfg["beta"]),
                beta_c=float(ompb_cfg["beta_c"]),
                delta=float(ompb_cfg["delta"]),
                c0=float(ompb_cfg["C0"]),
                device=device,
            )
            joblib.dump(constants, os.path.join(run_dir, "constants.joblib"))

            save_config_snapshot(
                run_dir,
                {
                    "ett": ett_cfg,
                    "ompb": ompb_cfg,
                    "models": models_cfg,
                    "args": args,
                    "backbone_src_dir": backbone_src_dir,
                },
            )
            print(f"[offline] Wrote {run_dir}")


if __name__ == "__main__":
    main()
