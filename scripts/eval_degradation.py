import os
import sys
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import glob as _glob
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.ett import ETTConfig, load_ett_datasets
from data.ili import ILIConfig, load_ili_datasets
from data.weather import WeatherConfig, load_weather_datasets
from scripts.train_backbone import build_backbone, eval_backbone_metrics
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


def main() -> None:

    args = parse_kv_args(sys.argv)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_ett_cfg, base_ompb_cfg, models_cfg = load_configs(root_dir, args.get("config_path"))
    # Optional runtime overrides (so pipeline.sh can control without editing YAML).
    backbone_batch_size = args.get("backbone_batch_size")
    tcn_batch_size = args.get("tcn_batch_size")
    autoformer_batch_size = args.get("autoformer_batch_size")
    eval_batch_size = _as_int(args.get("eval_batch_size"), 64)

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

    all_rows: List[Dict] = []

    def _metrics_first_n(backbone, dataset, n: int, batch_size: int = 64) -> tuple[float, float]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        y_true_all: List[np.ndarray] = []
        y_pred_all: List[np.ndarray] = []
        seen = 0
        for x, y in loader:
            with torch.no_grad():
                pred = backbone.predict_batch(x).detach().cpu().numpy()
            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(pred)
            seen += int(y.shape[0])
            if seen >= n:
                break
        y_true_np = np.concatenate(y_true_all, axis=0) if y_true_all else np.zeros((0,))
        y_pred_np = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.zeros((0,))
        y_true_np = y_true_np[:n]
        y_pred_np = y_pred_np[:n]
        mae = float(np.mean(np.abs(y_true_np - y_pred_np))) if y_true_np.size else 0.0
        mse = float(np.mean((y_true_np - y_pred_np) ** 2)) if y_true_np.size else 0.0
        return mae, mse

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

        set_seed(int(ompb_cfg.get("seed", 0)))

        device = ompb_cfg.get("device", "cpu")
        if isinstance(device, str) and device.startswith("cuda"):

            if not torch.cuda.is_available():
                print(f"CUDA requested ({device}) but not available; falling back to cpu")
                device = "cpu"

        if dataset == "ett":
            # Use split from config (do not override via CLI).
            split_cfg = dict(ett_cfg["data"]["split"])
            train_frac = float(split_cfg["train"])
            val_frac = float(split_cfg.get("val", 0.0))
            test_frac = float(split_cfg["test"])
            if train_frac <= 0.0 or test_frac <= 0.0 or val_frac < 0.0 or abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
                raise ValueError("ETT split must satisfy train>0, test>0, val>=0, and train+val+test==1.0")

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
            out_prefix = "degradation_backbone"
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
            out_prefix = "degradation_backbone_ili"
            metric_src_prefix = "ili_train_heldout"
            metric_tgt_prefix = "ili_test_match"
        else:
            # Load Weather config.
            # If pipeline passes config_path with a `weather:` section, use it; otherwise fall back to configs/weather.yaml.
            default_weather_path = os.path.join(root_dir, "configs", "weather.yaml")
            override_path = args.get("config_path")
            override = load_yaml(override_path) if override_path else None

            if override is not None and "weather" in override:
                weather_bundle = _deep_update(load_yaml(default_weather_path), {"weather": override["weather"]})
            else:
                weather_bundle = load_yaml(default_weather_path)

            # Allow ompb/models overrides from config_path even if it isn't a Weather config.
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
            out_prefix = "degradation_backbone_weather"
            metric_src_prefix = "weather_heldout"
            metric_tgt_prefix = "weather_close"

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
                "ili": None if dataset != "ili" else ili_cfg,
                "weather": weather_cfg if dataset == "weather" else None,
                "ompb": ompb_cfg,
                "models": models_cfg_local,
                "args": args,
                "n_test_windows": n_eval,
            },
        )

        for name in models_to_run:
            backbone = build_backbone(name, models_cfg_local, ompb_cfg, meta, device)

            # Try loading an existing checkpoint to skip training.
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
                # Save checkpoint for later reuse (model-specific filename).
                ckpt_path = os.path.join(out_dir, f"backbone_{name}.pt")
                if hasattr(backbone, "model") and isinstance(backbone.model, torch.nn.Module):
                    torch.save(backbone.model.state_dict(), ckpt_path)
                    print(f"  Backbone checkpoint saved → {ckpt_path}")
                else:
                    import joblib as _jl
                    _jl.dump(backbone, ckpt_path.replace(".pt", ".joblib"))
                    print(f"  Backbone checkpoint saved → {ckpt_path.replace('.pt', '.joblib')}")

            mae_src, mse_src = _metrics_first_n(backbone, ds_test, n=n_eval, batch_size=eval_batch_size)
            # Evaluate target on the same number of windows as the source heldout set.
            n_tgt = min(n_eval, len(ds_target))
            mae_tgt, mse_tgt = _metrics_first_n(backbone, ds_target, n=n_tgt, batch_size=eval_batch_size)

            row = {
                "model": name,
                "dataset": dataset,
                "seq_len": int(ompb_cfg["seq_len"]),
                "pred_len": int(pred_len),
                "train_frac": float(train_frac),
                "val_frac": float(val_frac),
                "test_frac": float(test_frac),
                "n_test_windows": int(n_eval),
                "n_target_windows": int(n_tgt),
                f"{metric_src_prefix}_mae": float(mae_src),
                f"{metric_src_prefix}_mse": float(mse_src),
                f"{metric_tgt_prefix}_mae": float(mae_tgt),
                f"{metric_tgt_prefix}_mse": float(mse_tgt),
                "delta_mae": float(mae_tgt - mae_src),
                "delta_mse": float(mse_tgt - mse_src),
                "run_dir": out_dir,
            }

            # Weather: also evaluate on the "far" test set.
            if dataset == "weather":
                n_far = min(n_eval, len(ds_target_far))
                mae_far, mse_far = _metrics_first_n(backbone, ds_target_far, n=n_far, batch_size=eval_batch_size)
                row["n_far_windows"] = int(n_far)
                row["weather_far_mae"] = float(mae_far)
                row["weather_far_mse"] = float(mse_far)
                row["delta_far_mae"] = float(mae_far - mae_src)
                row["delta_far_mse"] = float(mse_far - mse_src)

            all_rows.append(row)

            if dataset == "weather":
                print(
                    f"{name} pl={pred_len} {dataset}:heldout({n_eval}) MAE={mae_src:.4f} MSE={mse_src:.4f} | "
                    f"{dataset}:close({n_tgt}) MAE={mae_tgt:.4f} MSE={mse_tgt:.4f} | "
                    f"{dataset}:far({n_far}) MAE={row['weather_far_mae']:.4f} MSE={row['weather_far_mse']:.4f} | "
                    f"Δclose MAE={row['delta_mae']:.4f} MSE={row['delta_mse']:.4f} | "
                    f"Δfar MAE={row['delta_far_mae']:.4f} MSE={row['delta_far_mse']:.4f}"
                )
            else:
                print(
                    f"{name} pl={pred_len} {dataset}:heldout MAE={mae_src:.4f} MSE={mse_src:.4f} | "
                    f"{dataset}:target:first{n_eval} MAE={mae_tgt:.4f} MSE={mse_tgt:.4f} | "
                    f"ΔMAE={row['delta_mae']:.4f} ΔMSE={row['delta_mse']:.4f}"
                )

        save_csv(
            os.path.join(out_dir, "degradation.csv"),
            [r for r in all_rows if int(r["pred_len"]) == int(pred_len)],
        )


if __name__ == "__main__":
    main()

