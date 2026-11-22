"""OjinjinBench-v1 standardized evaluation harness.

Evaluates any AI-comment detector fairly on the same frozen JSONL benchmark.
Key metrics tailored for silent comment filtering (where false positives hurt most):
- PR-AUC (threshold-free ranking power under class imbalance)
- Recall @ Precision >= 99% (strict silent-filter ceiling)
- Recall @ Precision >= 95% (aggressive clean mode)
- Best F0.5 (precision weighted 2x higher than recall) & Best F1
- 4-bucket breakdown with 95% Wilson confidence intervals:
  obvious_ai, stealth_ai, hard_human, normal_human.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, target_prec: float) -> tuple[float, float, float]:
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    best_rec, best_prec, best_thr = 0.0, 1.0, 1.0
    for i in range(len(thr)):
        if prec[i] >= target_prec and rec[i] > best_rec:
            best_rec, best_prec, best_thr = float(rec[i]), float(prec[i]), float(thr[i])
    return best_rec, best_prec, best_thr


def best_f_beta(y_true: np.ndarray, y_score: np.ndarray, beta: float = 0.5) -> tuple[float, float, float, float]:
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    b2 = beta * beta
    best = (0.0, 0.0, 0.0, 0.5)
    for i in range(len(thr)):
        p, r = float(prec[i]), float(rec[i])
        denom = b2 * p + r
        fb = (1 + b2) * p * r / denom if denom > 0 else 0.0
        if fb > best[0]:
            best = (fb, p, r, float(thr[i]))
    return best


def evaluate_predictions(
    bench_items: list[dict[str, Any]],
    scores_by_id: dict[int, float],
    model_name: str = "model",
) -> dict[str, Any]:
    y_true_list: list[int] = []
    y_score_list: list[float] = []
    sub_buckets: list[str] = []
    missing = 0

    for idx, it in enumerate(bench_items):
        item_id = int(it.get("id", it.get("rpid", idx)))
        if item_id not in scores_by_id:
            missing += 1
            continue
        label = int(it["label"])
        y_true_list.append(label)
        y_score_list.append(float(scores_by_id[item_id]))

        rule_raw = int((it.get("rules") or {}).get("total_raw", it.get("rule_raw", 0)))
        if label == 1:
            sub = "obvious_ai" if rule_raw >= 60 else "stealth_ai"
        else:
            is_hard = (
                it.get("bucket") in ("hard_negative", "grey_zone")
                or rule_raw >= 20
                or int((it.get("rules") or {}).get("clean_len", len(it.get("message", "")))) >= 65
            )
            sub = "hard_human" if is_hard else "normal_human"
        sub_buckets.append(sub)

    y_true = np.asarray(y_true_list, dtype=np.int32)
    y_score = np.asarray(y_score_list, dtype=np.float64)
    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        raise ValueError("Benchmark must contain both positive (AI) and negative (human) samples.")

    pr_auc = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))
    rec99, prec99, thr99 = recall_at_precision(y_true, y_score, 0.99)
    rec95, prec95, thr95 = recall_at_precision(y_true, y_score, 0.95)
    f05, f05_p, f05_r, thr_f05 = best_f_beta(y_true, y_score, beta=0.5)
    f1, f1_p, f1_r, thr_f1 = best_f_beta(y_true, y_score, beta=1.0)

    def breakdown_at(thr: float) -> dict[str, Any]:
        out: dict[str, Any] = {}
        preds = (y_score >= thr).astype(np.int32)
        for name in ("obvious_ai", "stealth_ai", "hard_human", "normal_human"):
            idxs = [i for i, b in enumerate(sub_buckets) if b == name]
            n = len(idxs)
            if n == 0:
                continue
            hits = int(preds[idxs].sum())
            lo, hi = _wilson_ci(hits, n)
            if name.endswith("_ai"):
                out[name] = {"n": n, "recall": round(hits / n, 4), "ci95": [round(lo, 4), round(hi, 4)], "caught": hits}
            else:
                out[name] = {"n": n, "fpr": round(hits / n, 4), "ci95": [round(lo, 4), round(hi, 4)], "false_positives": hits}
        return out

    return {
        "model": model_name,
        "n_eval": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int(len(y_true) - y_true.sum()),
        "missing": missing,
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "recall_at_p99": {"recall": round(rec99, 4), "precision": round(prec99, 4), "threshold": round(thr99, 5)},
        "recall_at_p95": {"recall": round(rec95, 4), "precision": round(prec95, 4), "threshold": round(thr95, 5)},
        "best_f05": {"f05": round(f05, 4), "precision": round(f05_p, 4), "recall": round(f05_r, 4), "threshold": round(thr_f05, 5)},
        "best_f1": {"f1": round(f1, 4), "precision": round(f1_p, 4), "recall": round(f1_r, 4), "threshold": round(thr_f1, 5)},
        "breakdown_at_p95": breakdown_at(thr95),
        "breakdown_at_p99": breakdown_at(thr99),
    }


def format_leaderboard(reports: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | PR-AUC | Recall@P99% | Recall@P95% | Best F0.5 | Stealth AI Rec (@P95) | Hard Human FPR (@P95) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        b95 = r.get("breakdown_at_p95", {})
        st = b95.get("stealth_ai", {}).get("recall", 0.0)
        hh = b95.get("hard_human", {}).get("fpr", 0.0)
        lines.append(
            f"| {r['model']} | {r['pr_auc']:.4f} | "
            f"{r['recall_at_p99']['recall']*100:.1f}% | "
            f"{r['recall_at_p95']['recall']*100:.1f}% | "
            f"{r['best_f05']['f05']:.4f} | "
            f"{st*100:.1f}% | {hh*100:.2f}% |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction JSONLs on OjinjinBench-v1")
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--pred", action="append", required=True, help="name=path.jsonl (each line: {id/rpid, score})")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    bench_items = [json.loads(l) for l in args.bench.read_text(encoding="utf-8").splitlines() if l.strip()]
    reports = []
    for spec in args.pred:
        name, p_str = spec.split("=", 1)
        preds = {}
        for line in Path(p_str).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            preds[int(obj.get("id", obj["rpid"]))] = float(obj["score"])
        reports.append(evaluate_predictions(bench_items, preds, name))

    print(format_leaderboard(reports))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
