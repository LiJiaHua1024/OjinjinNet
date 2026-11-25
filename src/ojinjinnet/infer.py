"""Zero-PyTorch, zero-Transformers ONNX INT8 inference engine for OjinjinNet.

Runtime dependencies: ONLY `onnxruntime` and `numpy`.
Implements standard BERT Chinese WordPiece tokenization (`vocab.txt`) plus
Head+Tail truncation so opening hooks and closing AI sign-offs both survive.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import threading
import unicodedata
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


@lru_cache(maxsize=4096)
def _is_whitespace(ch: str) -> bool:
    return ch in (" ", "\t", "\n", "\r") or unicodedata.category(ch) == "Zs"


@lru_cache(maxsize=4096)
def _is_control(ch: str) -> bool:
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch) in ("Cc", "Cf")


@lru_cache(maxsize=4096)
def _is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


@lru_cache(maxsize=4096)
def _is_chinese_char(cp: int) -> bool:
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


@lru_cache(maxsize=4096)
def _strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


class WordPieceTokenizer:
    """Minimal BERT Chinese WordPiece tokenizer aligned with HuggingFace `BertTokenizer(do_lower_case=True)`."""

    def __init__(self, vocab_path: str | Path) -> None:
        # Match HuggingFace `BertTokenizer` exactly: split ONLY on '\n'/'\r\n', NOT Unicode linebreaks!
        self.vocab: dict[str, int] = {}
        with Path(vocab_path).open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                tok = line.rstrip("\r\n")
                self.vocab[tok] = idx
        self.pad_id = self.vocab.get("[PAD]", 0)
        self.unk_id = self.vocab.get("[UNK]", 100)
        self.cls_id = self.vocab.get("[CLS]", 101)
        self.sep_id = self.vocab.get("[SEP]", 102)
        self._wordpiece_cache: OrderedDict[str, list[int]] = OrderedDict()
        self._wordpiece_cache_lock = threading.Lock()
        self._wordpiece_cache_limit = 8192

    def _basic_tokenize(self, text: str) -> list[str]:
        # 1. Clean control chars & pad CJK chars with spaces
        buf: list[str] = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            if _is_whitespace(ch):
                buf.append(" ")
            elif _is_chinese_char(cp):
                buf.append(" ")
                buf.append(ch)
                buf.append(" ")
            else:
                buf.append(ch)
        cleaned = "".join(buf)

        # 2. Split on whitespace, lowercase, strip accents, split punctuation
        tokens: list[str] = []
        for raw in cleaned.strip().split():
            word = _strip_accents(raw.lower())
            start = 0
            for i, ch in enumerate(word):
                if _is_punctuation(ch):
                    if i > start:
                        tokens.append(word[start:i])
                    tokens.append(ch)
                    start = i + 1
            if start < len(word):
                tokens.append(word[start:])
        return tokens

    def _wordpiece_uncached(self, token: str) -> list[int]:
        if len(token) > 100:
            return [self.unk_id]
        sub_ids: list[int] = []
        start = 0
        n = len(token)
        while start < n:
            end = n
            found: int | None = None
            while start < end:
                sub = token[start:end] if start == 0 else "##" + token[start:end]
                tid = self.vocab.get(sub)
                if tid is not None:
                    found = tid
                    break
                end -= 1
            if found is None:
                return [self.unk_id]
            sub_ids.append(found)
            start = end
        return sub_ids

    def _wordpiece(self, token: str) -> list[int]:
        with self._wordpiece_cache_lock:
            cached = self._wordpiece_cache.get(token)
            if cached is not None:
                self._wordpiece_cache.move_to_end(token)
                return cached

        ids = self._wordpiece_uncached(token)
        with self._wordpiece_cache_lock:
            cached = self._wordpiece_cache.get(token)
            if cached is not None:
                self._wordpiece_cache.move_to_end(token)
                return cached
            self._wordpiece_cache[token] = ids
            if len(self._wordpiece_cache) > self._wordpiece_cache_limit:
                self._wordpiece_cache.popitem(last=False)
            return ids

    def _split_raw_token(self, raw: str) -> list[str]:
        word = _strip_accents(raw.lower())
        tokens: list[str] = []
        start = 0
        for i, ch in enumerate(word):
            if _is_punctuation(ch):
                if i > start:
                    tokens.append(word[start:i])
                tokens.append(ch)
                start = i + 1
        if start < len(word):
            tokens.append(word[start:])
        return tokens

    def encode(self, text: str, max_len: int = 192, head_len: int = 140) -> list[int]:
        ids: list[int] = []
        raw: list[str] = []
        vocab_get = self.vocab.get
        unk_id = self.unk_id

        def flush_raw() -> None:
            if not raw:
                return
            for tok in self._split_raw_token("".join(raw)):
                ids.extend(self._wordpiece(tok))
            raw.clear()

        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            if _is_whitespace(ch):
                flush_raw()
            elif _is_chinese_char(cp):
                flush_raw()
                ids.append(vocab_get(ch, unk_id))
            else:
                raw.append(ch)
        flush_raw()

        body_budget = max_len - 2
        if len(ids) > body_budget:
            tail_len = body_budget - head_len
            ids = ids[:head_len] + ids[-tail_len:]
        return [self.cls_id] + ids + [self.sep_id]


def _bucket_batches(
    seqs: list[list[int]],
    max_len: int,
    batch_size: int,
) -> list[tuple[list[int], list[list[int]]]]:
    buckets: dict[int, list[int]] = {}
    for idx, seq in enumerate(seqs):
        n = len(seq)
        key = 32 if n <= 32 else (64 if n <= 64 else (128 if n <= 128 else max_len))
        buckets.setdefault(key, []).append(idx)

    jobs: list[tuple[list[int], list[list[int]]]] = []
    for key in sorted(buckets):
        idxs = buckets[key]
        for i in range(0, len(idxs), batch_size):
            sub_idxs = idxs[i : i + batch_size]
            jobs.append((sub_idxs, [seqs[k] for k in sub_idxs]))
    return jobs


def _run_bucketed_items(
    batch_fn: Any,
    seqs: list[list[int]],
    max_len: int,
    batch_size: int,
    parallel: bool,
    executor: concurrent.futures.ThreadPoolExecutor | None,
) -> list[Any]:
    jobs = _bucket_batches(seqs, max_len=max_len, batch_size=batch_size)
    if parallel and len(jobs) > 1 and executor is not None:
        job_results = list(executor.map(lambda job: batch_fn(job[1]), jobs))
    else:
        job_results = [batch_fn(job_seqs) for _, job_seqs in jobs]

    output: list[Any] = [None] * len(seqs)
    for (idxs, _), rows in zip(jobs, job_results):
        for idx, val in zip(idxs, rows):
            output[idx] = val
    return output


class OjinjinClassifier:
    """Fast CPU ONNX INT8 classifier for detecting AI-generated Chinese comments."""

    def __init__(
        self,
        model_dir: str | Path,
        num_threads: int = 4,
        parallel_buckets: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        model_path = self.model_dir / "model_int8.onnx"
        vocab_path = self.model_dir / "vocab.txt"
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not vocab_path.is_file():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        self.tokenizer = WordPieceTokenizer(vocab_path)

        # Read sequence budget & optional fusion weights from manifest.json / fusion_weights.json
        self.max_len = 192
        self.head_len = 140
        manifest_path = self.model_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.max_len = int(manifest.get("max_len", self.max_len))
            self.head_len = int(manifest.get("head_len", 94 if self.max_len == 128 else 140))
        self.fusion: dict[str, Any] | None = None
        fw_path = self.model_dir / "fusion_weights.json"
        if fw_path.is_file():
            self.fusion = json.loads(fw_path.read_text(encoding="utf-8"))
            self.max_len = int(self.fusion.get("max_len", self.max_len))
            self.head_len = int(self.fusion.get("head_len", 94 if self.max_len == 128 else 140))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.parallel_buckets = parallel_buckets
        self._parallel_executor: concurrent.futures.ThreadPoolExecutor | None = None

    def _get_parallel_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._parallel_executor is None:
            workers = min(4, max(1, os.cpu_count() or 1))
            self._parallel_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="ojinjin-buckets",
            )
        return self._parallel_executor

    def close(self) -> None:
        if self._parallel_executor is not None:
            self._parallel_executor.shutdown(wait=True)
            self._parallel_executor = None

    def __enter__(self) -> "OjinjinClassifier":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _run_batch(self, seqs: list[list[int]]) -> np.ndarray:
        pad_id = self.tokenizer.pad_id
        ml = max(len(s) for s in seqs)
        ids = np.full((len(seqs), ml), pad_id, dtype=np.int64)
        mask = np.zeros((len(seqs), ml), dtype=np.int64)
        for j, seq in enumerate(seqs):
            ids[j, : len(seq)] = seq
            mask[j, : len(seq)] = 1
        return self.session.run(None, {"input_ids": ids, "attention_mask": mask})[0].reshape(-1).astype(np.float32)

    def predict_logits(
        self,
        texts: list[str],
        batch_size: int = 32,
        bucketed: bool = True,
        parallel_buckets: bool | None = None,
    ) -> np.ndarray:
        """Return temperature-calibrated log-odds for each comment text.

        When `bucketed=True`, groups sequences into compact length buckets (<=32 / <=64 / <=128 / <=max_len)
        so one long essay comment never forces short comments in the same batch to pad to 192 tokens
        (verified 2.2x-2.3x faster on real mixed-length comment pages with r>=0.999 numerical fidelity).
        """
        if not texts:
            return np.zeros((0,), dtype=np.float32)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        seqs = [self.tokenizer.encode(t, max_len=self.max_len, head_len=self.head_len) for t in texts]
        if not bucketed or len(seqs) <= 1:
            outs: list[np.ndarray] = []
            for i in range(0, len(seqs), batch_size):
                outs.append(self._run_batch(seqs[i : i + batch_size]))
            return np.concatenate(outs, axis=0)

        use_parallel = self.parallel_buckets if parallel_buckets is None else parallel_buckets
        rows = _run_bucketed_items(
            self._run_batch,
            seqs,
            max_len=self.max_len,
            batch_size=batch_size,
            parallel=use_parallel,
            executor=self._get_parallel_executor() if use_parallel else None,
        )
        return np.asarray(rows, dtype=np.float32)

    def predict_proba(
        self,
        texts: list[str],
        batch_size: int = 32,
        bucketed: bool = True,
        parallel_buckets: bool | None = None,
    ) -> list[float]:
        """Return calibrated AI-generated probability in [0, 1] using comment text only."""
        logits = self.predict_logits(
            texts,
            batch_size=batch_size,
            bucketed=bucketed,
            parallel_buckets=parallel_buckets,
        )
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return [float(p) for p in probs]

    def predict_with_fusion(self, text: str, metadata_features: dict[str, float]) -> float:
        """Optional end-to-end probability combining pure-text logit with lightweight metadata features."""
        if not self.fusion:
            return self.predict_proba([text])[0]
        logit = float(self.predict_logits([text])[0])
        names: list[str] = self.fusion["feature_names"]
        weights: list[float] = self.fusion["weights"]
        bias = float(self.fusion["bias"])
        vals = dict(metadata_features)
        vals["neural_logit"] = logit
        z = bias + sum(w * float(vals.get(k, 0.0)) for k, w in zip(names, weights))
        return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))


class OjinjinCascadeClassifier:
    """Two-stage CPU cascade routing (~90% handled by 1.22 MB Nano, ~10% escalated to 49.5 MB Pro).

    Supports two routing strategies:
      - `l2d` (default when `model_cascade_int8.onnx` is present): uses Nano's trained FP-aware
        Learning-to-Defer head (`defer_logits`) to route error-prone samples to the strong model.
      - `naive`: routes samples closest to `p = 0.5` (`0.5 - |p - 0.5| >= threshold`).
    """

    def __init__(
        self,
        fast_model_dir: str | Path,
        strong_model_dir: str | Path,
        strategy: str = "l2d",
        target_budget: int = 10,
        defer_threshold: float | None = None,
        num_threads: int = 4,
        parallel_buckets: bool = False,
    ) -> None:
        self.fast_dir = Path(fast_model_dir)
        self.strong = OjinjinClassifier(
            strong_model_dir,
            num_threads=num_threads,
            parallel_buckets=parallel_buckets,
        )
        self.fast = OjinjinClassifier(
            self.fast_dir,
            num_threads=num_threads,
            parallel_buckets=parallel_buckets,
        )
        self.parallel_buckets = parallel_buckets

        manifest_path = self.fast_dir / "manifest.json"
        self.manifest: dict[str, Any] = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        )
        cascade_onnx = self.fast_dir / str(self.manifest.get("cascade_model", "model_cascade_int8.onnx"))
        self.has_l2d_head = cascade_onnx.is_file()

        if strategy not in ("l2d", "naive"):
            raise ValueError(f"Unsupported cascade strategy: {strategy!r} (expected 'l2d' or 'naive')")
        self.strategy = "l2d" if (strategy == "l2d" and self.has_l2d_head) else "naive"

        self.cascade_session: ort.InferenceSession | None = None
        if self.strategy == "l2d":
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = num_threads
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.cascade_session = ort.InferenceSession(
                str(cascade_onnx), sess_options=opts, providers=["CPUExecutionProvider"]
            )

        if defer_threshold is not None:
            self.defer_threshold = float(defer_threshold)
        else:
            thrs = self.manifest.get("cascade_thresholds") or {}
            key = "l2d_fp" if self.strategy == "l2d" else "naive"
            bucket_map = thrs.get(key) or {}
            bkey = str(int(target_budget))
            if bkey in bucket_map:
                self.defer_threshold = float(bucket_map[bkey])
            else:
                self.defer_threshold = 0.25 if self.strategy == "l2d" else 0.12

    def _run_fast_both(self, texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        if self.cascade_session is None:
            logits = self.fast.predict_logits(texts, batch_size=batch_size, bucketed=True)
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            return logits, (0.5 - np.abs(probs - 0.5)).astype(np.float32)

        seqs = [self.fast.tokenizer.encode(t, max_len=self.fast.max_len, head_len=self.fast.head_len) for t in texts]
        pad_id = self.fast.tokenizer.pad_id

        def run_cascade_batch(sub_seqs: list[list[int]]) -> list[tuple[float, float]]:
            ml = max(len(s) for s in sub_seqs)
            ids = np.full((len(sub_seqs), ml), pad_id, dtype=np.int64)
            mask = np.zeros((len(sub_seqs), ml), dtype=np.int64)
            for j, s in enumerate(sub_seqs):
                ids[j, : len(s)] = s
                mask[j, : len(s)] = 1
            z, d = self.cascade_session.run(None, {"input_ids": ids, "attention_mask": mask})
            z_flat = z.reshape(-1).astype(np.float32)
            d_flat = d.reshape(-1).astype(np.float32)
            d_prob = 1.0 / (1.0 + np.exp(-np.clip(d_flat, -30.0, 30.0)))
            return list(zip(z_flat, d_prob))

        rows = _run_bucketed_items(
            run_cascade_batch,
            seqs,
            max_len=self.fast.max_len,
            batch_size=batch_size,
            parallel=self.parallel_buckets,
            executor=self.fast._get_parallel_executor() if self.parallel_buckets else None,
        )
        out_z = np.asarray([row[0] for row in rows], dtype=np.float32)
        out_d = np.asarray([row[1] for row in rows], dtype=np.float32)
        return out_z, out_d

    def close(self) -> None:
        self.fast.close()
        self.strong.close()

    def __enter__(self) -> "OjinjinCascadeClassifier":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def predict_detailed(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        """Return per-comment calibrated probability, escalation flag, and defer score."""
        if not texts:
            return []
        fast_z, defer_scores = self._run_fast_both(texts, batch_size=batch_size)
        fast_p = 1.0 / (1.0 + np.exp(-np.clip(fast_z, -30.0, 30.0)))
        escalate_idxs = [i for i, d in enumerate(defer_scores) if float(d) >= self.defer_threshold]

        final_p = fast_p.astype(np.float64).copy()
        escalated_set = set(escalate_idxs)
        if escalate_idxs:
            strong_probs = self.strong.predict_proba([texts[i] for i in escalate_idxs], batch_size=batch_size, bucketed=True)
            for idx, sp in zip(escalate_idxs, strong_probs):
                final_p[idx] = sp

        return [
            {
                "prob": float(final_p[i]),
                "escalated": i in escalated_set,
                "defer_score": float(defer_scores[i]),
                "fast_prob": float(fast_p[i]),
            }
            for i in range(len(texts))
        ]

    def predict_proba(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """Return calibrated AI-generated probability in [0, 1] using two-stage cascade routing."""
        return [row["prob"] for row in self.predict_detailed(texts, batch_size=batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OjinjinNet ONNX INT8 inference on Chinese comments.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory with model_int8.onnx and vocab.txt")
    parser.add_argument(
        "--cascade-strong-dir",
        type=Path,
        default=None,
        help="Optional strong model dir (e.g. Pro) to enable two-stage Nano->Pro cascade routing",
    )
    parser.add_argument("--cascade-strategy", choices=["l2d", "naive"], default="l2d", help="Cascade routing strategy")
    parser.add_argument("--cascade-budget", type=int, choices=[10, 15, 20], default=10, help="Target escalation budget (%)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for ONNX Runtime")
    parser.add_argument(
        "--parallel-buckets",
        action="store_true",
        help="Run length-bucketed ONNX batches concurrently (mainly useful for tiny models such as Nano)",
    )
    parser.add_argument("texts", nargs="*", help="Comment strings to score (or pipe lines via stdin)")
    args = parser.parse_args()

    texts = list(args.texts)
    if not texts and not sys.stdin.isatty():
        texts = [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        parser.error("Provide at least one comment string as argument or via stdin.")

    if args.cascade_strong_dir is not None:
        cas = OjinjinCascadeClassifier(
            args.model_dir,
            args.cascade_strong_dir,
            strategy=args.cascade_strategy,
            target_budget=args.cascade_budget,
            num_threads=args.threads,
            parallel_buckets=args.parallel_buckets,
        )
        for text, row in zip(texts, cas.predict_detailed(texts)):
            tag = "AI" if row["prob"] >= 0.5 else "Human"
            route = "Pro" if row["escalated"] else "Fast"
            print(f"[{tag:5s}|{route:4s}] p={row['prob']:.4f} (fast={row['fast_prob']:.4f}, defer={row['defer_score']:.3f}) | {text}")
    else:
        clf = OjinjinClassifier(
            args.model_dir,
            num_threads=args.threads,
            parallel_buckets=args.parallel_buckets,
        )
        probs = clf.predict_proba(texts)
        for text, p in zip(texts, probs):
            tag = "AI" if p >= 0.5 else "Human"
            print(f"[{tag:5s}] p={p:.4f} | {text}")


if __name__ == "__main__":
    main()
