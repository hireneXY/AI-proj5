#!/usr/bin/env python3
"""
Lightweight experiment runner to sweep MISA-related configs and collect results.

Features:
- grid over reconstruction/similarity/difference weights (cartesian)
- toggle text_cleaning on/off
- runs experiments sequentially, writes logs to results/logs/, temp configs to tmp_configs/
- aggregates validation JSON metrics into a summary CSV

Usage (example):
  python3 scripts/run_experiments.py --models misa --recon 0,0.05,0.1 --sim 0,0.05,0.1 --diff 0,0.05 \
    --epochs 8 --batch_size 2 --accumulation_steps 8 --lr 1e-4
"""
import os
import argparse
import yaml
import subprocess
import time
import glob
import json
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP_DIR = os.path.join(ROOT, "tmp_configs")
LOG_DIR = os.path.join(ROOT, "results", "logs")
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def write_tmp_config(base_yaml, extra, fname):
    cfg = {}
    # merge base if exists
    if base_yaml and os.path.exists(base_yaml):
        with open(base_yaml, "r", encoding="utf-8") as f:
            try:
                cfg = yaml.safe_load(f) or {}
            except Exception:
                cfg = {}
    # shallow merge training.loss_weights and top-level text_cleaning_enable or augmentation.text
    training = cfg.get("training", {})
    loss_weights = training.get("loss_weights", {})
    loss_weights.update(extra.get("loss_weights", {}))
    training["loss_weights"] = loss_weights
    cfg["training"] = training
    # text cleaning
    if "text_cleaning_enable" in extra:
        cfg["text_cleaning_enable"] = bool(extra["text_cleaning_enable"])
    # class weights
    if "class_weights" in extra:
        cfg.setdefault("training", {}).setdefault("class_weights", {})
        cfg["training"]["class_weights"].update(extra["class_weights"])

    path = os.path.join(TMP_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path

def run_main(cmd_args, log_path):
    with open(log_path, "w", encoding="utf-8") as lf:
        print("RUN:", " ".join(cmd_args))
        proc = subprocess.run(cmd_args, stdout=lf, stderr=subprocess.STDOUT)
    return proc.returncode

def aggregate_results(out_csv):
    files = sorted(glob.glob(os.path.join(ROOT, "results", "*val_metrics_epoch*.json")))
    rows = []
    for f in files:
        try:
            j = json.load(open(f, "r", encoding="utf-8"))
            rows.append({
                "file": os.path.basename(f),
                "model": j.get("model_type"),
                "epoch": j.get("epoch"),
                "acc": j.get("accuracy"),
                "f1_weighted": j.get("f1_weighted"),
                "loss": j.get("loss")
            })
        except Exception:
            continue
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["file","model","epoch","acc","f1_weighted","loss"])
        for r in rows:
            w.writerow([r["file"], r["model"], r["epoch"], r["acc"], r["f1_weighted"], r["loss"]])
    return out_csv

def parse_list_arg(s):
    if not s:
        return []
    return [float(x) for x in s.split(",") if x.strip()!=""]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default="misa", help="comma separated model names")
    p.add_argument("--recon", type=str, default="0.0,0.05,0.1,0.2")
    p.add_argument("--sim", type=str, default="0.0,0.05,0.1,0.2")
    p.add_argument("--diff", type=str, default="0.0,0.05,0.1")
    p.add_argument("--text_clean", action="store_true", help="if set, also run text_clean on variants (otherwise off)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--base_config", type=str, default=os.path.join(ROOT,"multimodal_config.yaml"))
    p.add_argument("--use_amp", action="store_true")
    args = p.parse_args()

    recon_vals = parse_list_arg(args.recon)
    sim_vals = parse_list_arg(args.sim)
    diff_vals = parse_list_arg(args.diff)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    runs = []
    for model in models:
        for r in recon_vals:
            for s in sim_vals:
                for d in diff_vals:
                    for clean in ([True, False] if args.text_clean else [True]):
                        name = f"{model}_rec{r}_sim{s}_diff{d}_clean{int(clean)}"
                        runs.append((model, r, s, d, clean, name))

    print(f"Prepared {len(runs)} runs. Temp configs in {TMP_DIR}, logs in {LOG_DIR}")
    for model, r, s, d, clean, name in runs:
        extra = {"loss_weights": {"reconstruction": r, "similarity": s, "difference": d},
                 "text_cleaning_enable": bool(clean)}
        tmp_name = f"{name}.yaml"
        tmp_path = write_tmp_config(args.base_config, extra, tmp_name)
        log_path = os.path.join(LOG_DIR, f"{name}.log")
        cmd = ["python3", os.path.join(ROOT,"main.py"),
               "--model", model,
               "--epochs", str(args.epochs),
               "--batch_size", str(args.batch_size),
               "--accumulation_steps", str(args.accumulation_steps),
               "--lr", str(args.lr),
               "--data_dir", args.data_dir,
               "--config", tmp_path
               ]
        if args.use_amp:
            cmd.append("--use_amp")
        rc = run_main(cmd, log_path)
        print(f"Finished {name} (rc={rc}), log={log_path}")
        time.sleep(3)

    summary = aggregate_results(os.path.join(ROOT,"results","exp_summary_clean_grid.csv"))
    print("Wrote summary:", summary)

if __name__ == "__main__":
    main()


