import logging
from typing import Dict, Optional, Any, List

try:
    from transformers import pipeline
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None
    torch = None

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    def __init__(
        self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ):
        self.model_name = model_name
        self.classifier = None

        if not TRANSFORMERS_AVAILABLE:
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
            )
        except Exception as e:
            logger.warning(f"HuggingFace não disponível: {e}")
            self.classifier = None

    def classify_email(self, email_text: str) -> Optional[Dict[str, Any]]:
        if not self.classifier:
            return None

        if not email_text or not email_text.strip():
            return None

        try:
            results = self.classifier(email_text)
            return self._parse_pipeline_results(results)
        except Exception as e:
            logger.error(f"Erro na classificação HuggingFace: {e}")
            return None

    def _parse_pipeline_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results or not results[0]:
            return self._get_default_result()

        best_result = max(results[0], key=lambda x: x["score"])

        label = best_result["label"].lower()
        confidence = best_result["score"]

        if any(
            word in label for word in ["positive", "pos", "5", "4", "excellent", "good"]
        ):
            category = "produtivo"
        elif any(
            word in label for word in ["negative", "neg", "1", "2", "bad", "terrible"]
        ):
            category = "improdutivo"
        elif "neutral" in label or "3" in label:
            category = "produtivo" if confidence > 0.6 else "improdutivo"
        else:
            category = "produtivo" if confidence > 0.6 else "improdutivo"

        return {
            "category": category,
            "confidence": confidence,
            "method": "huggingface",
            "reasoning": f"Classificação Hugging Face: {best_result['label']} ({confidence:.2f})",
        }

    def _get_default_result(self) -> Dict[str, Any]:
        return {
            "category": "improdutivo",
            "confidence": 0.5,
            "method": "huggingface",
            "reasoning": "Erro na classificação Hugging Face, usando fallback",
        }

    def is_available(self) -> bool:
        return self.classifier is not None
