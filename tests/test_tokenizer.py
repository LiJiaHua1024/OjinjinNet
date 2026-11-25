"""Regression tests for the optimized WordPiece tokenizer."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from ojinjinnet.infer import (
    WordPieceTokenizer,
    _is_chinese_char,
    _is_control,
    _is_punctuation,
    _is_whitespace,
    _strip_accents,
)


class ReferenceWordPieceTokenizer(WordPieceTokenizer):
    """The pre-optimization tokenization pipeline, kept as an exact reference."""

    def _basic_tokenize_reference(self, text: str) -> list[str]:
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

    def _wordpiece_reference(self, token: str) -> list[int]:
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

    def encode_reference(self, text: str, max_len: int = 192, head_len: int = 140) -> list[int]:
        ids: list[int] = []
        for tok in self._basic_tokenize_reference(text):
            ids.extend(self._wordpiece_reference(tok))
        body_budget = max_len - 2
        if len(ids) > body_budget:
            tail_len = body_budget - head_len
            ids = ids[:head_len] + ids[-tail_len:]
        return [self.cls_id] + ids + [self.sep_id]


class WordPieceTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        vocab_tokens = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            *list("的一是不了人我在有他这中大来上国个地说们为子和你出道教时年得就那要下以生会自着去之过家学对可她哈好真笑看声"),
            *list("abcdefghijklmnopqrstuvwxyz0123456789"),
            *list("##abcdefghijklmnopqrstuvwxyz0123456789"),
            *list("，。！？；：、“”‘’（）《》【】—…·!?,.()[]{}"),
            "café",
            "naïve",
            "über",
        ]
        cls._tmp = tempfile.TemporaryDirectory()
        vocab_path = Path(cls._tmp.name) / "vocab.txt"
        vocab_path.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
        cls.tok = WordPieceTokenizer(vocab_path)
        cls.ref = ReferenceWordPieceTokenizer(vocab_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def assert_same_encoding(self, text: str) -> None:
        for max_len, head_len in ((192, 140), (128, 94), (32, 20)):
            self.assertEqual(
                self.ref.encode_reference(text, max_len=max_len, head_len=head_len),
                self.tok.encode(text, max_len=max_len, head_len=head_len),
                msg=f"text={text!r}, max_len={max_len}",
            )

    def test_representative_edge_cases(self) -> None:
        cases = [
            "",
            "哈",
            "哈哈哈这个视频真的太有意思了，看完直接笑出声哈哈哈",
            "不得不说，这个up主的分析真的很到位！good job!!!",
            "Hello世界123😂😂，Café NAÏVE Über test",
            "ab\x00cd\x1fef\ngh\r\nijk\ttuv  xyz",
            "，。！？；：、“”‘’（）《》【】—…·",
            "𠀀𠀁𠀂汉字test",
            "a" * 150,
            "哈" * 300,
            "AI生成的评论通常会使用非常工整的结构，而且喜欢进行总结。",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assert_same_encoding(text)

    def test_deterministic_random_unicode_mix(self) -> None:
        rng = random.Random(0)
        alphabet = "的一是不了人我在有他这中大来上国个地说们为子和你abcXYZ0123，。！？!?,.😂éü \n\t\x00\x1f"
        for _ in range(200):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 180)))
            with self.subTest(text=text):
                self.assert_same_encoding(text)


if __name__ == "__main__":
    unittest.main()
