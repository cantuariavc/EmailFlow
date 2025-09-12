import logging
from typing import Dict, Optional, Any, List
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """
    Cliente para classificação de texto usando modelos do Hugging Face
    """

    def __init__(
        self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ):
        self.model_name = model_name
        self.classifier = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._load_model()
            logger.info(
                f"Modelo Hugging Face carregado: {model_name} (device: {self.device})"
            )
        except Exception as e:
            logger.error(f"Erro ao carregar modelo Hugging Face: {e}")
            self.classifier = None

    def _load_model(self):
        """Carrega o modelo de classificação"""
        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                top_k=None,
            )
            logger.info("Pipeline de classificação carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar pipeline: {e}")
            self.classifier = None

    def classify_email(self, email_text: str) -> Optional[Dict[str, Any]]:
        """
        Classifica um email usando Hugging Face
        """
        if not self.is_available():
            return None

        if not email_text or not email_text.strip():
            logger.warning("Texto do email vazio para classificação Hugging Face")
            return None

        try:
            if self.classifier:
                results = self.classifier(email_text)
                return self._parse_pipeline_results(results)
            else:
                logger.warning("Pipeline Hugging Face não disponível")
                return self._get_default_result()

        except Exception as e:
            logger.error(f"Erro na classificação Hugging Face: {e}")
            return None

    def _parse_pipeline_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processa resultados do pipeline"""
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
        """Retorna resultado padrão em caso de erro"""
        return {
            "category": "improdutivo",
            "confidence": 0.5,
            "method": "huggingface",
            "reasoning": "Erro na classificação Hugging Face, usando fallback",
        }

    def is_available(self) -> bool:
        """Verifica se o cliente Hugging Face está disponível"""
        return self.classifier is not None
