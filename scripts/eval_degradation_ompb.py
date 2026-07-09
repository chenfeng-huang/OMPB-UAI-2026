import os
import sys
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import glob as _glob
import joblib
import numpy as np
import torch

from data.ett import ETTConfig, load_ett_datasets
from data.ili import ILIConfig, load_ili_datasets
from data.weather import WeatherConfig, load_weather_datasets
from models.bayesian_head import BayesianLinearHead
from ompb.bound import compute_variance_proxy_constants
from ompb.online_calibration import run_online_calibration
from scripts.train_offline import _sample_windows, _train_head, build_backbone
from utils.config import _deep_update, apply_cli_overrides, load_configs, load_yaml
from utils.logging import make_run_dir, save_config_snapshot, save_csv
from utils.seed import set_seed


def parse_kv_args(argv) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for arg in argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            parsed[k] = v
    return parsed


def _as_float(v: Optional[str], default: float) -> float:
    return default if v is None else float(v)


def _as_int(v: Optional[str], default: int) -> int:
    return default if v is None else int(v)


def _find_latest_backbone_ckpt(
    log_dir: str, seq_len: int, pred_len: int, train_frac: float, model_name: str,
) -> Optional[str]:
    """Scan *log_dir* for the most recent backbone checkpoint matching the config."""
    pat = os.path.join(
        log_dir,
        f"*_sl{seq_len}_pl{pred_len}_tr{train_frac:.2f}_*",
        f"backbone_{model_name}.pt",
    )
    hits = sorted(_glob.glob(pat))
    return hits[-1] if hits else None


class _ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


def main() -> None:

    args = parse_kv_args(sys.argv)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_ett_cfg, base_ompb_cfg, models_cfg = load_configs(root_dir, args.get("config_path"))
    # Optional runtime overrides (so pipeline.sh can control without editing YAML).
    backbone_batch_size = args.get("backbone_batch_size")
    tcn_batch_size = args.get("tcn_batch_size")
    autoformer_batch_size = args.get("autoformer_batch_size")
    src_batch_size = args.get("src_batch_size")
    prefetch_target = args.get("prefetch_target")

    # Support sweep mode via pred_lens=24,48,96,... (comma-separated).
    if "pred_lens" in args:
        pred_lens = [int(x) for x in args["pred_lens"].split(",") if x.strip() != ""]
        if not pred_lens:
            raise ValueError("pred_lens was provided but empty. Example: pred_lens=24,48,96")
    else:
        pred_lens = [int(args["pred_len"])] if "pred_len" in args else [int(base_ompb_cfg["pred_len"])]

    dataset = args.get("dataset", "ett").strip().lower()
    if dataset not in ("ett", "ili", "weather"):
        raise ValueError("dataset must be one of: ett|ili|weather")

    model_name = args.get("model", "all")
    models_filter = args.get("models")
    max_n = args.get("max_n")
    max_n_int = None if max_n is None else int(max_n)

    retrain = args.get("retrain", "0") not in ("0", "false", "False", "no", "No")
    backbone_dir = args.get("backbone_dir", "").strip() or None

    # Offline head training controls
    head_train_n = args.get("head_train_N")  
    head_epochs = _as_int(args.get("head_epochs"), 20)
    head_batch_size = _as_int(args.get("head_batch_size"), 256)

    # OMPB controls
    update_rate = args.get("update_rate")
    cal_steps = args.get("cal_steps_J")
    cal_lr = args.get("cal_lr")
    buffer_l = args.get("buffer_L")
    progress = args.get("progress")
    sigma0 = args.get("sigma0")
    alpha_init = args.get("alpha_init")
    alpha_prior = args.get("alpha_prior")
    lambda_sup = args.get("lambda_sup")

    default_models = ["tcn", "autoformer", "gpt4ts"]
    if model_name != "all":
        if models_filter is not None:
            raise ValueError("Use models=... only with model=all")
        models_to_run = [model_name]
    else:
        if models_filter is None:
            models_to_run = default_models
        else:
            models_to_run = [m.strip() for m in models_filter.split(",") if m.strip() != ""]
            unknown = [m for m in models_to_run if m not in default_models]
            if unknown:
                raise ValueError(f"Unknown models in models=...: {unknown}. Allowed: {default_models}")

    rows: List[Dict] = []

    for pred_len in pred_lens:
        ett_cfg = copy.deepcopy(base_ett_cfg)
        ompb_cfg = copy.deepcopy(base_ompb_cfg)
        models_cfg_local = copy.deepcopy(models_cfg)
        # Override backbone training batch sizes if provided.
        if backbone_batch_size is not None:
            bs = int(backbone_batch_size)
            models_cfg_local.setdefault("tcn", {})["batch_size"] = bs
            models_cfg_local.setdefault("autoformer", {})["batch_size"] = bs
        if tcn_batch_size is not None:
            models_cfg_local.setdefault("tcn", {})["batch_size"] = int(tcn_batch_size)
        if autoformer_batch_size is not None:
            models_cfg_local.setdefault("autoformer", {})["batch_size"] = int(autoformer_batch_size)

        local_args = dict(args)
        local_args["pred_len"] = str(pred_len)
        apply_cli_overrides(local_args, ett_cfg, ompb_cfg)

        if dataset == "ett":
            split_cfg = dict(ett_cfg["data"]["split"])
            train_frac = float(split_cfg["train"])
            val_frac = float(split_cfg.get("val", 0.0))
            test_frac = float(split_cfg["test"])
            if train_frac <= 0.0 or test_frac <= 0.0 or val_frac < 0.0 or abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
                raise ValueError("ETT split must satisfy train>0, test>0, val>=0, and train+val+test==1.0")

        # Apply OMPB overrides
        if update_rate is not None:
            ompb_cfg["update_rate"] = float(update_rate)
        if cal_steps is not None:
            ompb_cfg["cal_steps_J"] = int(cal_steps)
        if cal_lr is not None:
            ompb_cfg["cal_lr"] = float(cal_lr)
        if buffer_l is not None:
            ompb_cfg["buffer_L"] = int(buffer_l)
        if progress is not None:
            ompb_cfg["progress"] = int(progress)
        if sigma0 is not None:
            ompb_cfg["sigma0"] = float(sigma0)
        if alpha_init is not None:
            ompb_cfg["alpha_init"] = float(alpha_init)
        if alpha_prior is not None:
            ompb_cfg["alpha_prior"] = float(alpha_prior)
        if lambda_sup is not None:
            ompb_cfg["lambda_sup"] = float(lambda_sup)
        if src_batch_size is not None:
            ompb_cfg["src_batch_size"] = int(src_batch_size)
        if prefetch_target is not None:
            ompb_cfg["prefetch_target"] = int(prefetch_target)

        set_seed(int(ompb_cfg.get("seed", 0)))

        device = str(ompb_cfg.get("device", "cpu"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"CUDA requested ({device}) but not available; falling back to cpu")
            device = "cpu"

        if dataset == "ett":
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
            ds_train = datasets["etth1"]["train"]
            ds_test = datasets["etth1"]["test"]
            ds_target = datasets["etth2"]["test"]
            ds_target_far = None
            out_prefix = "degradation_ompb"
            metric_src_prefix = "etth1_test"
            metric_tgt_prefix = "etth2_match"
        elif dataset == "ili":
            # Load ILI config.
            # If pipeline passes config_path with an `ili:` section, use it; otherwise fall back to configs/ili.yaml.
            default_ili_path = os.path.join(root_dir, "configs", "ili.yaml")
            override_path = args.get("config_path")
            override = load_yaml(override_path) if override_path else None

            if override is not None and "ili" in override:
                ili_bundle = _deep_update(load_yaml(default_ili_path), {"ili": override["ili"]})
            else:
                ili_bundle = load_yaml(default_ili_path)

            # Allow ompb/models overrides from config_path even if it isn't an ILI config.
            if override is not None:
                if "ompb" in override:
                    ompb_cfg = _deep_update(ompb_cfg, override["ompb"])
                if "models" in override:
                    models_cfg_local = _deep_update(models_cfg_local, override["models"])

            ili_cfg = ili_bundle["ili"]

            split_cfg = dict(ili_cfg["data"]["split"])
            train_frac = float(split_cfg["train"])
            val_frac = float(split_cfg.get("val", 0.0))
            test_frac = float(split_cfg["test"])
            if train_frac <= 0.0 or test_frac <= 0.0 or val_frac < 0.0 or abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
                raise ValueError("ILI split must satisfy train>0, test>0, val>=0, and train+val+test==1.0")

            # For ILI, seq_len is defined in configs/ili.yaml (do not override from CLI).
            seq_len = int(ili_cfg["window"]["seq_len"])
            ompb_cfg["seq_len"] = seq_len
            ompb_cfg["pred_len"] = int(pred_len)

            ili = ILIConfig(
                train_path=str(ili_cfg["data"]["train_path"]),
                test_path=str(ili_cfg["data"]["test_path"]),
                split=split_cfg,
                seq_len=seq_len,
                pred_len=int(pred_len),
                scaler_path=str(ili_cfg["scaling"]["scaler_path"]),
                align_columns=str(ili_cfg["data"].get("align_columns", "train")),
                impute_nan=bool(ili_cfg["scaling"].get("impute_nan", True)),
            )
            datasets = load_ili_datasets(ili)
            meta = datasets["meta"]
            ds_train = datasets["ili_train"]["train"]
            ds_test = datasets["ili_train"]["test"]
            ds_target = datasets["ili_test"]["test"]
            ds_target_far = None
            out_prefix = "degradation_ompb_ili"
            metric_src_prefix = "ili_train_heldout"
            metric_tgt_prefix = "ili_test_match"

        elif dataset == "weather":
            # Load Weather config.
            default_weather_path = os.path.join(root_dir, "configs", "weather.yaml")
            override_path = args.get("config_path")
            override = load_yaml(override_path) if override_path else None

            if override is not None and "weather" in override:
                weather_bundle = _deep_update(load_yaml(default_weather_path), {"weather": override["weather"]})
            else:
                weather_bundle = load_yaml(default_weather_path)

            if override is not None:
                if "ompb" in override:
                    ompb_cfg = _deep_update(ompb_cfg, override["ompb"])
                if "models" in override:
                    models_cfg_local = _deep_update(models_cfg_local, override["models"])

            weather_cfg = weather_bundle["weather"]

            split_cfg = dict(weather_cfg["data"]["split"])
            train_frac = float(split_cfg["train"])
            val_frac = float(split_cfg.get("val", 0.0))
            test_frac = float(split_cfg["test"])
            if train_frac <= 0.0 or test_frac <= 0.0 or val_frac < 0.0 or abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
                raise ValueError("Weather split must satisfy train>0, test>0, val>=0, and train+val+test==1.0")

            seq_len = int(weather_cfg["window"]["seq_len"])
            ompb_cfg["seq_len"] = seq_len
            ompb_cfg["pred_len"] = int(pred_len)

            weather = WeatherConfig(
                train_path=str(weather_cfg["data"]["train_path"]),
                test_close_path=str(weather_cfg["data"]["test_close_path"]),
                test_far_path=str(weather_cfg["data"]["test_far_path"]),
                split=split_cfg,
                seq_len=seq_len,
                pred_len=int(pred_len),
                scaler_path=str(weather_cfg["scaling"]["scaler_path"]),
                align_columns=str(weather_cfg["data"].get("align_columns", "strict")),
                impute_nan=bool(weather_cfg["scaling"].get("impute_nan", True)),
            )
            datasets = load_weather_datasets(weather)
            meta = datasets["meta"]
            ds_train = datasets["weather_train"]["train"]
            ds_test = datasets["weather_train"]["test"]
            ds_target = datasets["weather_test_close"]["test"]
            ds_target_far = datasets["weather_test_far"]["test"]
            out_prefix = "degradation_ompb_weather"
            metric_src_prefix = "weather_heldout"
            metric_tgt_prefix = "weather_close"

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        n_eval = len(ds_test)
        if max_n_int is not None:
            n_eval = min(n_eval, max_n_int)
        if n_eval <= 0:
            raise ValueError("Heldout test windows are empty for the chosen split/seq_len/pred_len.")

        out_dir = make_run_dir(
            ompb_cfg["log_dir"],
            f"{out_prefix}_sl{int(ompb_cfg['seq_len'])}_pl{pred_len}_tr{train_frac:.2f}",
        )
        save_config_snapshot(
            out_dir,
            {
                "ett": ett_cfg,
                "ili": ili_cfg if dataset == "ili" else None,
                "weather": weather_cfg if dataset == "weather" else None,
                "ompb": ompb_cfg,
                "models": models_cfg_local,
                "args": args,
                "n_eval_windows": n_eval,
            },
        )

        for name in models_to_run:
            # Train (or load) backbone
            backbone = build_backbone(name, models_cfg_local, ompb_cfg, meta, device)

            loaded = False
            if not retrain:
                if backbone_dir:
                    _ckpt = os.path.join(backbone_dir, f"backbone_{name}.pt")
                else:
                    _ckpt = _find_latest_backbone_ckpt(
                        ompb_cfg["log_dir"], int(ompb_cfg["seq_len"]),
                        pred_len, train_frac, name,
                    )
                if _ckpt and os.path.isfile(_ckpt):
                    backbone.model.load_state_dict(
                        torch.load(_ckpt, map_location=device)
                    )
                    print(f"  Loaded backbone checkpoint ← {_ckpt}")
                    loaded = True

            if not loaded:
                backbone.fit(ds_train)
                # Save checkpoint for later reuse.
                ckpt_path = os.path.join(out_dir, f"backbone_{name}.pt")
                if hasattr(backbone, "model") and isinstance(backbone.model, torch.nn.Module):
                    torch.save(backbone.model.state_dict(), ckpt_path)
                    print(f"  Backbone checkpoint saved → {ckpt_path}")
                else:
                    _jl_path = ckpt_path.replace(".pt", ".joblib")
                    joblib.dump(backbone, _jl_path)
                    print(f"  Backbone checkpoint saved → {_jl_path}")

            # Train PMPB head on ETTh1:train windows
            source_buf_n = int(ompb_cfg.get("source_buf_N", 256))
            if head_train_n is None:
                head_train = len(ds_train)
            else:
                head_train = int(head_train_n)
            x_train, y_train = _sample_windows(ds_train, n=head_train, seed=int(ompb_cfg.get("seed", 0)))

            head_base = BayesianLinearHead(
                pred_len=int(ompb_cfg["pred_len"]),
                out_channels=int(meta.get("y_channels", 1)),
                sigma0=float(ompb_cfg["sigma0"]),
                alpha_init=float(ompb_cfg.get("alpha_init", -6.0)),
                alpha_prior=float(ompb_cfg.get("alpha_prior", -6.0)),
            ).to(device)
            _train_head(
                head=head_base,
                backbone=backbone,
                x_train=x_train,
                y_train=y_train,
                device=device,
                lr=float(ompb_cfg["train_head_lr"]),
                epochs=head_epochs,
                batch_size=head_batch_size,
                lambda_pb=float(ompb_cfg.get("lambda_pb", 1.0)),
            )

            # Save trained head checkpoint.
            head_ckpt_path = os.path.join(out_dir, f"head_{name}.pt")
            torch.save(head_base.state_dict(), head_ckpt_path)
            print(f"  Head checkpoint saved → {head_ckpt_path}")

            # Source buffer + constants
            rng = np.random.default_rng(int(ompb_cfg.get("seed", 0)))
            buf_n = min(source_buf_n, x_train.shape[0])
            buf_idxs = rng.choice(x_train.shape[0], size=buf_n, replace=False)
            source_x = x_train[buf_idxs].astype(np.float32)
            source_y = y_train[buf_idxs].astype(np.float32)

            constants = compute_variance_proxy_constants(
                source_dataset=_ArrayDataset(source_x, source_y),
                backbone=backbone,
                head=head_base,
                alpha=float(ompb_cfg["alpha"]),
                beta=float(ompb_cfg["beta"]),
                beta_c=float(ompb_cfg["beta_c"]),
                delta=float(ompb_cfg["delta"]),
                c0=float(ompb_cfg["C0"]),
                device=device,
            )

            # Run OMPB separately on source heldout and target datasets using identical starting head.

            def _run(target_dataset, max_t: int):
                cfg_run = copy.deepcopy(ompb_cfg)
                cfg_run["max_t"] = max_t
                head = BayesianLinearHead(
                    pred_len=int(cfg_run["pred_len"]),
                    out_channels=int(meta.get("y_channels", 1)),
                    sigma0=float(cfg_run["sigma0"]),
                    alpha_init=float(cfg_run.get("alpha_init", -6.0)),
                    alpha_prior=float(cfg_run.get("alpha_prior", -6.0)),
                ).to(device)
                # Non-strict for backward compatibility if head design evolves.
                head.load_state_dict(head_base.state_dict(), strict=False)
                logs = run_online_calibration(
                    backbone=backbone,
                    head=head,
                    source_buffer=torch.from_numpy(source_x),
                    source_labels=torch.from_numpy(source_y),
                    target_dataset=target_dataset,
                    config=cfg_run,
                    constants=constants,
                    device=device,
                )
                mae = float(np.mean([r["mae"] for r in logs])) if logs else 0.0
                mse = float(np.mean([r["mse"] for r in logs])) if logs else 0.0
                return mae, mse

            mae_src, mse_src = _run(ds_test, max_t=int(n_eval))
            n_tgt = min(n_eval, len(ds_target))
            mae_tgt, mse_tgt = _run(ds_target, max_t=int(n_tgt))

            row = {
                "model": name,
                "dataset": dataset,
                "seq_len": int(ompb_cfg["seq_len"]),
                "pred_len": int(pred_len),
                "train_frac": float(train_frac),
                "val_frac": float(val_frac),
                "test_frac": float(test_frac),
                "n_eval_windows": int(n_eval),
                f"{metric_src_prefix}_mae": mae_src,
                f"{metric_src_prefix}_mse": mse_src,
                f"{metric_tgt_prefix}_mae": mae_tgt,
                f"{metric_tgt_prefix}_mse": mse_tgt,
                "delta_mae": float(mae_tgt - mae_src),
                "delta_mse": float(mse_tgt - mse_src),
                "out_dir": out_dir,
            }

            # Weather has a second (far) target.
            if ds_target_far is not None:
                n_far = min(n_eval, len(ds_target_far))
                mae_far, mse_far = _run(ds_target_far, max_t=int(n_far))
                row["n_tgt_windows"] = int(n_tgt)
                row["n_far_windows"] = int(n_far)
                row["weather_far_mae"] = mae_far
                row["weather_far_mse"] = mse_far
                row["delta_far_mae"] = float(mae_far - mae_src)
                row["delta_far_mse"] = float(mse_far - mse_src)

            rows.append(row)

            if ds_target_far is not None:
                print(
                    f"{name} pl={pred_len} OMPB {dataset}:heldout MAE={mae_src:.4f} MSE={mse_src:.4f} | "
                    f"{dataset}:close({n_tgt}) MAE={mae_tgt:.4f} MSE={mse_tgt:.4f} | "
                    f"{dataset}:far({n_far}) MAE={row['weather_far_mae']:.4f} MSE={row['weather_far_mse']:.4f} | "
                    f"Δclose MAE={row['delta_mae']:.4f} MSE={row['delta_mse']:.4f} | "
                    f"Δfar MAE={row['delta_far_mae']:.4f} MSE={row['delta_far_mse']:.4f}"
                )
            else:
                print(
                    f"{name} pl={pred_len} OMPB {dataset}:heldout MAE={mae_src:.4f} MSE={mse_src:.4f} | "
                    f"{dataset}:target({n_tgt}) MAE={mae_tgt:.4f} MSE={mse_tgt:.4f} | "
                    f"ΔMAE={row['delta_mae']:.4f} ΔMSE={row['delta_mse']:.4f}"
                )

        save_csv(os.path.join(out_dir, "degradation_ompb.csv"), [r for r in rows if int(r["pred_len"]) == int(pred_len)])


if __name__ == "__main__":
    main()

