import os
import sys
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.ett import ETTConfig, load_ett_datasets
# NOTE: Keep heavyweight / optional deps out of import-time to make it possible
# to run "tcn only" experiments without requiring statsmodels, etc.
from models.autoformer_backbone import AutoformerBackbone
from models.gpt4ts_backbone import GPT4TSBackbone
from models.tcn_backbone import TCNBackbone
from utils.config import apply_cli_overrides, load_configs
from utils.logging import make_run_dir, save_config_snapshot, save_csv
from utils.seed import set_seed

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it


def parse_kv_args(argv) -> Dict[str, str]:
    parsed = {}
    for arg in argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            parsed[k] = v
    return parsed


def _select_eval_dataset(datasets: Dict, eval_dataset: str):
    if eval_dataset == "none":
        return None, None
    if eval_dataset == "etth2":
        return datasets["etth2"]["test"], "etth2:test"
    if eval_dataset == "etth1_train":
        return datasets["etth1"]["train"], "etth1:train"
    if eval_dataset == "etth1_val":
        return datasets["etth1"]["val"], "etth1:val"
    if eval_dataset == "etth1_test":
        return datasets["etth1"]["test"], "etth1:test"
    raise ValueError(f"Unknown eval_dataset={eval_dataset}. Use etth2|etth1_train|etth1_val|etth1_test|none")


def eval_backbone_metrics(backbone, dataset, batch_size: int) -> tuple[float, float]:
    """
    Evaluate a backbone on a window dataset.

    Note: backbone.predict_batch() implementations already handle any internal device moves and
    return CPU tensors (by convention in this repo), so we keep metric computation on CPU.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    y_true_all = []
    y_pred_all = []
    for x, y in loader:
        with torch.no_grad():
            y_pred = backbone.predict_batch(x)
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(y_pred.detach().cpu().numpy())

    y_true_np = np.concatenate(y_true_all, axis=0) if y_true_all else np.zeros((0,))
    y_pred_np = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.zeros((0,))
    mae = float(np.mean(np.abs(y_true_np - y_pred_np))) if y_true_np.size else 0.0
    mse = float(np.mean((y_true_np - y_pred_np) ** 2)) if y_true_np.size else 0.0
    return mae, mse


def build_backbone(model_name: str, cfg_models: Dict, cfg_ompb: Dict, meta: Dict, device: str):
    if model_name == "tcn":
        return TCNBackbone(
            in_channels=int(meta.get("x_channels", len(meta["feature_names"]))),
            out_channels=int(meta.get("y_channels", 1)),
            pred_len=cfg_ompb["pred_len"],
            widths=cfg_models["tcn"]["widths"],
            kernel_size=cfg_models["tcn"]["kernel_size"],
            dropout=cfg_models["tcn"]["dropout"],
            lr=cfg_models["tcn"]["lr"],
            epochs=cfg_models["tcn"]["epochs"],
            batch_size=cfg_models["tcn"]["batch_size"],
            device=device,
        )
    if model_name == "autoformer":
        return AutoformerBackbone(
            n_features=int(meta.get("x_channels", len(meta["feature_names"]))),
            out_channels=int(meta.get("y_channels", 1)),
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


def save_backbone_checkpoint(backbone, model_name: str, run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    if model_name in ["tcn", "autoformer", "gpt4ts"]:
        torch.save(backbone.model.state_dict(), os.path.join(run_dir, "backbone.pt"))
    else:
        joblib.dump(backbone, os.path.join(run_dir, "backbone.joblib"))


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
    eval_dataset = args.get("eval_dataset", "etth2")
    save_eval = args.get("save_eval", "1") == "1"
    eval_batch_size = int(args.get("eval_batch_size", args.get("batch_size", 64)))

    sweep_rows = []

    for pred_len in pred_lens:
        ett_cfg = copy.deepcopy(base_ett_cfg)
        ompb_cfg = copy.deepcopy(base_ompb_cfg)
        local_args = dict(args)
        local_args["pred_len"] = str(pred_len)

        # For ETTh sweeps we keep seq_len fixed (default 96 in configs). Users may still override explicitly.
        apply_cli_overrides(local_args, ett_cfg, ompb_cfg)
        set_seed(ompb_cfg["seed"])

        device = ompb_cfg.get("device", "cpu")
        if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
            print(f"CUDA requested ({device}) but not available; falling back to cpu")
            device = "cpu"

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
        eval_ds, eval_tag = _select_eval_dataset(datasets, eval_dataset)

        models_to_train = [model_name] if model_name != "all" else ["tcn", "autoformer", "gpt4ts"]
        for name in tqdm(models_to_train, desc=f"Training backbones (pred_len={pred_len})"):
            run_dir = make_run_dir(ompb_cfg["log_dir"], f"backbone_{name}_sl{ett_cfg['window']['seq_len']}_pl{pred_len}")
            save_config_snapshot(run_dir, {"ett": ett_cfg, "ompb": ompb_cfg, "models": models_cfg})

            backbone = build_backbone(name, models_cfg, ompb_cfg, meta, device)
            backbone.fit(train_dataset)
            save_backbone_checkpoint(backbone, name, run_dir)

            if eval_ds is not None:
                mae, mse = eval_backbone_metrics(backbone, eval_ds, batch_size=eval_batch_size)
                print(f"backbone {name} pred_len={pred_len} eval {eval_tag} MAE={mae:.4f} MSE={mse:.4f} run_dir={run_dir}")
                row = {
                    "model": name,
                    "seq_len": int(ett_cfg["window"]["seq_len"]),
                    "pred_len": int(pred_len),
                    "eval": eval_tag,
                    "mae": mae,
                    "mse": mse,
                    "run_dir": run_dir,
                }
                sweep_rows.append(row)
                if save_eval:
                    save_csv(
                        os.path.join(run_dir, f"eval_{eval_tag.replace(':', '_')}.csv"),
                        [row],
                    )

    # Write one combined sweep summary (useful when looping many pred_lens)
    if sweep_rows and save_eval:
        sweep_dir = make_run_dir(base_ompb_cfg["log_dir"], "sweep_backbone")
        save_config_snapshot(sweep_dir, {"ett": base_ett_cfg, "ompb": base_ompb_cfg, "models": models_cfg, "args": args})
        save_csv(os.path.join(sweep_dir, "sweep_metrics.csv"), sweep_rows)

        # Also write aggregated (pivoted) CSVs: rows=models, columns=pred_len.
        # If multiple eval tags are present, write one set per eval tag.
        pred_lens_sorted = sorted({int(r["pred_len"]) for r in sweep_rows})
        eval_tags = sorted({str(r.get("eval", "")) for r in sweep_rows})
        for eval_tag in eval_tags:
            rows_for_eval = [r for r in sweep_rows if str(r.get("eval", "")) == eval_tag]
            if not rows_for_eval:
                continue
            models_sorted = sorted({str(r["model"]) for r in rows_for_eval})

            def _pivot(metric: str) -> list[dict]:
                out_rows: list[dict] = []
                for m in models_sorted:
                    row = {"model": m}
                    for pl in pred_lens_sorted:
                        key = f"pl{pl}"
                        match = next((x for x in rows_for_eval if x["model"] == m and int(x["pred_len"]) == pl), None)
                        row[key] = "" if match is None else match.get(metric, "")
                    out_rows.append(row)
                return out_rows

            tag = eval_tag.replace(":", "_") if eval_tag else "eval"
            save_csv(os.path.join(sweep_dir, f"agg_{tag}_mae.csv"), _pivot("mae"))
            save_csv(os.path.join(sweep_dir, f"agg_{tag}_mse.csv"), _pivot("mse"))


if __name__ == "__main__":
    main()
