"""Zero-PyTorch, zero-Transformers ONNX INT8 inference engine for OjinjinNet.

Runtime dependencies: ONLY `onnxruntime` and `numpy`.
Implements standard BERT Chinese WordPiece tokenization (`vocab.txt`) plus
Head+Tail truncation so opening hooks and closing AI sign-offs both survive.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def _is_whitespace(ch: str) -> bool:
    return ch in (" ", "\t", "\n", "\r") or unicodedata.category(ch) == "Zs"


def _is_control(ch: str) -> bool:
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch) in ("Cc", "Cf")


def _is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


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

    def _wordpiece(self, token: str) -> list[int]:
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

    def encode(self, text: str, max_len: int = 192, head_len: int = 140) -> list[int]:
        ids: list[int] = []
        for tok in self._basic_tokenize(text):
            ids.extend(self._wordpiece(tok))
        body_budget = max_len - 2
        if len(ids) > body_budget:
            tail_len = body_budget - head_len
            ids = ids[:head_len] + ids[-tail_len:]
        return [self.cls_id] + ids + [self.sep_id]


class OjinjinClassifier:
    """Fast CPU ONNX INT8 classifier for detecting AI-generated Chinese comments."""

    def __init__(
        self,
        model_dir: str | Path,
        num_threads: int = 4,
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
    ) -> np.ndarray:
        """Return temperature-calibrated log-odds for each comment text.

        When `bucketed=True`, groups sequences into compact length buckets (<=32 / <=64 / <=128 / <=max_len)
        so one long essay comment never forces short comments in the same batch to pad to 192 tokens
        (verified 2.2x-2.3x faster on real mixed-length comment pages with r>=0.999 numerical fidelity).
        """
        if not texts:
            return np.zeros((0,), dtype=np.float32)
        seqs = [self.tokenizer.encode(t, max_len=self.max_len, head_len=self.head_len) for t in texts]
        if not bucketed or len(seqs) <= 1:
            outs: list[np.ndarray] = []
            for i in range(0, len(seqs), batch_size):
                outs.append(self._run_batch(seqs[i : i + batch_size]))
            return np.concatenate(outs, axis=0)

        buckets: dict[int, list[int]] = {}
        for idx, seq in enumerate(seqs):
            n = len(seq)
            key = 32 if n <= 32 else (64 if n <= 64 else (128 if n <= 128 else self.max_len))
            buckets.setdefault(key, []).append(idx)

        result = np.empty((len(seqs),), dtype=np.float32)
        for key in sorted(buckets):
            idxs = buckets[key]
            for i in range(0, len(idxs), batch_size):
                sub_idxs = idxs[i : i + batch_size]
                sub_logits = self._run_batch([seqs[k] for k in sub_idxs])
                for k, val in zip(sub_idxs, sub_logits):
                    result[k] = val
        return result

    def predict_proba(
        self,
        texts: list[str],
        batch_size: int = 32,
        bucketed: bool = True,
    ) -> list[float]:
        """Return calibrated AI-generated probability in [0, 1] using comment text only."""
        logits = self.predict_logits(texts, batch_size=batch_size, bucketed=bucketed)
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
    ) -> None:
        self.fast_dir = Path(fast_model_dir)
        self.strong = OjinjinClassifier(strong_model_dir, num_threads=num_threads)
        self.fast = OjinjinClassifier(self.fast_dir, num_threads=num_threads)

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
        n = len(seqs)
        out_z = np.empty((n,), dtype=np.float32)
        out_d = np.empty((n,), dtype=np.float32)
        buckets: dict[int, list[int]] = {}
        for idx, seq in enumerate(seqs):
            l = len(seq)
            key = 32 if l <= 32 else (64 if l <= 64 else (128 if l <= 128 else self.fast.max_len))
            buckets.setdefault(key, []).append(idx)
        for key in sorted(buckets):
            idxs = buckets[key]
            for i in range(0, len(idxs), batch_size):
                sub = idxs[i : i + batch_size]
                sub_seqs = [seqs[k] for k in sub]
                ml = max(len(s) for s in sub_seqs)
                ids = np.full((len(sub), ml), pad_id, dtype=np.int64)
                mask = np.zeros((len(sub), ml), dtype=np.int64)
                for j, s in enumerate(sub_seqs):
                    ids[j, : len(s)] = s
                    mask[j, : len(s)] = 1
                z, d = self.cascade_session.run(None, {"input_ids": ids, "attention_mask": mask})
                z_flat = z.reshape(-1).astype(np.float32)
                d_flat = d.reshape(-1).astype(np.float32)
                d_prob = 1.0 / (1.0 + np.exp(-np.clip(d_flat, -30.0, 30.0)))
                for k, zv, dv in zip(sub, z_flat, d_prob):
                    out_z[k] = zv
                    out_d[k] = dv
        return out_z, out_d

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
        )
        for text, row in zip(texts, cas.predict_detailed(texts)):
            tag = "AI" if row["prob"] >= 0.5 else "Human"
            route = "Pro" if row["escalated"] else "Fast"
            print(f"[{tag:5s}|{route:4s}] p={row['prob']:.4f} (fast={row['fast_prob']:.4f}, defer={row['defer_score']:.3f}) | {text}")
    else:
        clf = OjinjinClassifier(args.model_dir, num_threads=args.threads)
        probs = clf.predict_proba(texts)
        for text, p in zip(texts, probs):
            tag = "AI" if p >= 0.5 else "Human"
            print(f"[{tag:5s}] p={p:.4f} | {text}")


if __name__ == "__main__":
    main()
