"""OjinjinNet: Ultra-lightweight on-device Chinese AI-comment detector (ONNX INT8)."""

from .infer import OjinjinClassifier, WordPieceTokenizer

__all__ = ["OjinjinClassifier", "WordPieceTokenizer"]
__version__ = "1.1.0"
