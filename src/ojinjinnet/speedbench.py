"""Deterministic CPU inference speed benchmark for OjinjinNet models."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .infer import OjinjinClassifier

PAGE_TOKEN_LENS = [32] * 12 + [64] * 5 + [128] * 3
CJK_CHARS = "的一是不了人我在有他这中大来上国个地说们为子和你出道教时年得就那要下以生会自着去之过家学对可她里后小么心多到天而好然无"


def synthetic_page(seed: int = 0) -> list[str]:
    """Return a deterministic 20-comment page matching the documented 12x32+5x64+3x128 distribution."""
    rng = random.Random(seed)
    return [
        "".join(rng.choice(CJK_CHARS) for _ in range(token_len - 2))
        for token_len in PAGE_TOKEN_LENS
    ]


def time_call(fn: Callable[[], Any], reps: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "mean_ms": float(np.mean(arr)),
    }


def bench_one(
    model_dir: str | Path,
    num_threads: int,
    reps: int,
    warmup: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    texts = synthetic_page(seed)
    clf = OjinjinClassifier(model_dir, num_threads=num_threads, parallel_buckets=False)
    try:
        seq_lens = sorted({len(clf.tokenizer.encode(t, max_len=clf.max_len, head_len=clf.head_len)) for t in texts})
        tokenize = time_call(
            lambda: [clf.tokenizer.encode(t, max_len=clf.max_len, head_len=clf.head_len) for t in texts],
            reps=reps,
            warmup=warmup,
        )

        serial_logits = clf.predict_logits(texts, batch_size=batch_size, parallel_buckets=False)
        parallel_logits = clf.predict_logits(texts, batch_size=batch_size, parallel_buckets=True)
        max_logit_diff = float(np.max(np.abs(serial_logits - parallel_logits))) if len(serial_logits) else 0.0

        serial = time_call(
            lambda: clf.predict_proba(texts, batch_size=batch_size, parallel_buckets=False),
            reps=reps,
            warmup=warmup,
        )
        parallel = time_call(
            lambda: clf.predict_proba(texts, batch_size=batch_size, parallel_buckets=True),
            reps=reps,
            warmup=warmup,
        )
    finally:
        clf.close()

    return {
        "threads": num_threads,
        "n_comments": len(texts),
        "sequence_lengths": seq_lens,
        "tokenize_page": tokenize,
        "serial_page": serial,
        "parallel_page": parallel,
        "serial_comments_per_sec": len(texts) * 1000.0 / serial["median_ms"],
        "parallel_comments_per_sec": len(texts) * 1000.0 / parallel["median_ms"],
        "speedup": serial["median_ms"] / parallel["median_ms"],
        "max_logit_diff": max_logit_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OjinjinNet ONNX CPU inference throughput.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory with model_int8.onnx and vocab.txt")
    parser.add_argument("--threads", default="1,2,4", help="Comma-separated ONNX Runtime thread counts")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    thread_counts = [int(x) for x in args.threads.split(",") if x.strip()]
    reports = [
        bench_one(
            args.model_dir,
            num_threads=n,
            reps=args.reps,
            warmup=args.warmup,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        for n in thread_counts
    ]

    print("| Threads | Tokenize p50 ms | Serial p50 ms | Parallel p50 ms | Speedup | Parallel comments/s | Max logit diff |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for r in reports:
        print(
            f"| {r['threads']} | {r['tokenize_page']['median_ms']:.3f} | "
            f"{r['serial_page']['median_ms']:.3f} | {r['parallel_page']['median_ms']:.3f} | "
            f"{r['speedup']:.2f}x | {r['parallel_comments_per_sec']:.0f} | {r['max_logit_diff']:.3g} |"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
