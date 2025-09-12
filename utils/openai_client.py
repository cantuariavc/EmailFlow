import openai
import logging
from typing import Dict, Optional, Any
from config import Config

logger = logging.getLogger(__name__)
CONFIDENCE_HIGH = 0.9
CONFIDENCE_LOW = 0.6

SYSTEM_PROMPT_CLASSIFICATION = "Você é um especialista em classificação de emails corporativos. Seja preciso e conciso."
SYSTEM_PROMPT_RESPONSE = (
    "Você é um assistente corporativo. Seja conciso e profissional."
)

CLASSIFICATION_PROMPT = """Classifique este email como PRODUTIVO ou IMPRODUTIVO:

Email: {email_text}

Responda apenas: PRODUTIVO ou IMPRODUTIVO"""

RESPONSE_PROMPT_PRODUCTIVE = """Gere uma resposta profissional para este email produtivo:

Email: {email_text}

Responda de forma concisa confirmando que será processado."""

RESPONSE_PROMPT_GENERAL = """Gere uma resposta educada para este email:

Email: {email_text}

Responda de forma breve e profissional."""


class OpenAIClient:
    """
    Cliente simplificado para integração com a API da OpenAI GPT
    """

    def __init__(self):
        self.client = None
        self.config = Config()

        if self.config.OPENAI_API_KEY:
            try:
                self.client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
                logger.info("Cliente OpenAI inicializado com sucesso!")
            except Exception as e:
                logger.error(f"Erro ao inicializar cliente OpenAI: {e}")
                self.client = None
        else:
            logger.warning(
                "Chave da API OpenAI não encontrada. Usando classificação baseada em regras."
            )

    def classify_email(self, email_text: str) -> Optional[Dict[str, Any]]:
        """
        Classifica um email usando a API da OpenAI
        """
        if not self.client:
            return None

        if not email_text or not email_text.strip():
            logger.warning("Texto do email vazio para classificação")
            return None

        if len(email_text.strip()) < self.config.MIN_TEXT_LENGTH:
            logger.warning(
                f"Texto muito curto para classificação: {len(email_text)} caracteres"
            )
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_CLASSIFICATION},
                    {
                        "role": "user",
                        "content": CLASSIFICATION_PROMPT.format(email_text=email_text),
                    },
                ],
                max_tokens=self.config.OPENAI_MAX_TOKENS,
                temperature=self.config.OPENAI_TEMPERATURE,
                timeout=30,
            )

            result_text = response.choices[0].message.content
            if result_text:
                return self._parse_classification_response(result_text.strip())
            return None

        except openai.RateLimitError as e:
            logger.error(f"Rate limit excedido na classificação: {e}")
            return None
        except openai.APIConnectionError as e:
            logger.error(f"Erro de conexão com OpenAI na classificação: {e}")
            return None
        except openai.APIError as e:
            if "insufficient_quota" in str(e):
                logger.error(
                    f"Quota insuficiente na OpenAI. Desabilitando OpenAI e usando classificação baseada em regras: {e}"
                )
            else:
                logger.error(f"Erro da API OpenAI na classificação: {e}")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado na classificação OpenAI: {e}")
            return None

    def generate_response(
        self, email_text: str, classification: Dict[str, Any]
    ) -> Optional[str]:
        """
        Gera uma resposta usando a API da OpenAI
        """
        if not self.client:
            return None

        try:
            category = classification.get("category", "improdutivo")

            if category == "produtivo":
                prompt = RESPONSE_PROMPT_PRODUCTIVE.format(email_text=email_text)
            else:
                prompt = RESPONSE_PROMPT_GENERAL.format(email_text=email_text)

            response = self.client.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_RESPONSE},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.OPENAI_MAX_TOKENS * 2,
                temperature=0.5,
                timeout=30,
            )

            result_text = response.choices[0].message.content
            return result_text.strip() if result_text else None

        except openai.RateLimitError as e:
            logger.error(f"Rate limit excedido na geração de resposta: {e}")
            return None
        except openai.APIConnectionError as e:
            logger.error(f"Erro de conexão com OpenAI na geração de resposta: {e}")
            return None
        except openai.APIError as e:
            if "insufficient_quota" in str(e):
                logger.error(
                    f"Quota insuficiente na OpenAI. Desabilitando OpenAI e usando templates de resposta: {e}"
                )
            else:
                logger.error(f"Erro da API OpenAI na geração de resposta: {e}")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado na geração de resposta OpenAI: {e}")
            return None

    def _parse_classification_response(self, response_text: str) -> Dict[str, Any]:
        """
        Processa a resposta da OpenAI de forma simplificada
        """
        text = response_text.lower().strip()

        if "produtivo" in text:
            return {
                "category": "produtivo",
                "confidence": CONFIDENCE_HIGH,
                "reasoning": f"Classificação: {response_text}",
            }
        elif "improdutivo" in text:
            return {
                "category": "improdutivo",
                "confidence": CONFIDENCE_HIGH,
                "reasoning": f"Classificação: {response_text}",
            }
        else:
            return {
                "category": "improdutivo",
                "confidence": CONFIDENCE_LOW,
                "reasoning": f"Classificação ambígua: {response_text}",
            }

    def is_available(self) -> bool:
        """Verifica se o cliente OpenAI está disponível"""
        return self.client is not None
