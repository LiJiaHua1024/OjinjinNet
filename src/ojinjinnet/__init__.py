"""OjinjinNet: Ultra-lightweight on-device Chinese AI-comment detector (ONNX INT8)."""

from .infer import OjinjinCascadeClassifier, OjinjinClassifier, WordPieceTokenizer

__all__ = ["OjinjinCascadeClassifier", "OjinjinClassifier", "WordPieceTokenizer"]
__version__ = "1.3.0"
