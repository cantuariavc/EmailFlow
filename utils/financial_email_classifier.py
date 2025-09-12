import re
import logging
from typing import Dict, List, Any
from .openai_client import OpenAIClient
from .huggingface_client import HuggingFaceClient
from .nlp_utils import preprocess_text
from config import Config

logger = logging.getLogger(__name__)
config = Config()
MIN_CONFIDENCE_OPENAI = 0.7
MIN_CONFIDENCE_RESPONSE = 0.8
MIN_RESPONSE_LENGTH = 20
PRODUTIVO_THRESHOLD = 10.0
IMPRODUTIVO_THRESHOLD = 5.0


class FinancialEmailClassifier:
    """
    Classificador simplificado e otimizado para emails do setor financeiro
    """

    def __init__(self):
        self.openai_client = OpenAIClient()
        self.huggingface_client = (
            HuggingFaceClient(config.HUGGINGFACE_MODEL)
            if config.HUGGINGFACE_ENABLED
            else None
        )

        self.produtivo_patterns = [
            r"\b(status|situação|andamento|progresso|atualização|atualizar)\b",
            r"\b(requisição|solicitação|pedido|demanda|solicitar|requerer)\b",
            r"\b(processo|protocolo|ticket|chamado|processar)\b",
            r"\b(problema|erro|bug|falha|não\s+funciona|dificuldade)\b",
            r"\b(documento|arquivo|anexo|comprovante|certificado)\b",
            r"\b(aprovar|rejeitar|validar|confirmar|autorizar)\b",
            r"\b(urgente|emergência|crítico|prioridade|importante)\b",
            r"\b(compliance|regulamentação|auditoria|conformidade)\b",
            r"\b(suporte|assistência|orientação)\b",
            r"\b(empréstimo|financiamento|crédito|cartão)\b",
            r"\b(pagamento|cobrança|fatura|boleto)\b",
            r"\b(cadastro|informação|dados\s+da\s+conta)\b",
            r"\b(relatório|extrato|demonstrativo)\b",
            r"\b(reunião|agendamento|compromisso)\b",
            r"\b(preciso\s+de|necessito|gostaria\s+de)\b",
            r"\b(quando\s+será|quando\s+posso|quando\s+está)\b",
            r"\b(por\s+favor|favor\s+verificar)\b",
            r"\b(solicitação\s+de\s+crédito|pedido\s+de\s+crédito|crédito\s+solicitado)\b",
            r"\b(abri\s+na\s+semana|abri\s+ontem|abri\s+hoje|abri\s+na\s+quinta|abri\s+na\s+sexta)\b",
            r"\b(já\s+há|já\s+existe|já\s+tem|já\s+foi)\b",
            r"\b(alguma\s+atualização|atualização|novidade|informação)\b",
            r"\b(referente\s+à|sobre\s+a|relacionado\s+à|sobre\s+o)\b",
            r"\b(saber\s+se|gostaria\s+de\s+saber|quero\s+saber|preciso\s+saber)\b",
        ]

        self.improdutivo_patterns = [
            r"\b(feliz\s+natal|boas\s+férias|feliz\s+ano\s+novo|felicitações)\b",
            r"\b(parabéns|felicitações|comemoração|celebração)\b",
            r"\b(olá\s*$|oi\s*$|bom\s+dia\s*$|boa\s+tarde\s*$|boa\s+noite\s*$)\b",
            r"\b(promoção|oferta|desconto|cupom|marketing)\b",
            r"\b(pessoal|particular|privado|familiar)\b",
            r"\b(conversa|bate-papo|fofoca|rumor)\b",
            r"\b(como\s+vai|tudo\s+bem|espero\s+que\s+esteja\s+bem)\b",
            r"\b(apenas\s+para\s+dizer|só\s+para\s+cumprimentar|só\s+passando)\b",
            r"\b(divulgando|compartilhando|curtir|seguir)\b",
            r"\b(redes\s+sociais|facebook|instagram|whatsapp)\b",
        ]

        self.response_templates = {
            "produtivo": "Obrigado pelo contato. Sua solicitação foi registrada e está sendo processada por nossa equipe. Retornaremos em até 24 horas úteis com as informações solicitadas.",
            "improdutivo": "Obrigado pela mensagem. Caso tenha alguma solicitação específica relacionada aos nossos serviços, estarei à disposição para ajudar.",
        }

    def _classify_by_rules(self, email_text: str) -> Dict[str, Any]:
        """Classificação baseada em padrões melhorada"""
        email_lower = email_text.lower().strip()
        produtivo_matches = sum(
            len(re.findall(pattern, email_lower, re.IGNORECASE))
            for pattern in self.produtivo_patterns
        )
        improdutivo_matches = sum(
            len(re.findall(pattern, email_lower, re.IGNORECASE))
            for pattern in self.improdutivo_patterns
        )

        total_words = len(email_lower.split())
        produtivo_score = (produtivo_matches / max(total_words, 1)) * 100
        improdutivo_score = (improdutivo_matches / max(total_words, 1)) * 100

        if produtivo_score > PRODUTIVO_THRESHOLD:
            category = "produtivo"
            confidence = min(0.95, 0.7 + (produtivo_score - improdutivo_score) * 0.01)
        elif improdutivo_score > IMPRODUTIVO_THRESHOLD:
            category = "improdutivo"
            confidence = min(0.95, 0.7 + (improdutivo_score - produtivo_score) * 0.01)
        else:
            if produtivo_score > improdutivo_score:
                category = "produtivo"
                confidence = 0.6
            else:
                category = "improdutivo"
                confidence = 0.6

        return {
            "category": category,
            "confidence": confidence,
            "method": "rules",
            "reasoning": f"Padrões encontrados: {produtivo_matches} produtivos, {improdutivo_matches} improdutivos",
        }

    def generate_response(
        self, processed_text: str, classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera uma resposta automática simplificada
        """
        category = classification["category"]

        if (
            self.openai_client.is_available()
            and classification.get("method") == "openai"
            and classification.get("confidence", 0) > MIN_CONFIDENCE_RESPONSE
        ):
            try:
                openai_response = self.openai_client.generate_response(
                    processed_text, classification
                )
                if (
                    openai_response
                    and len(openai_response.strip()) > MIN_RESPONSE_LENGTH
                ):
                    return {
                        "response": openai_response,
                        "suggested_actions": self._suggest_actions(category),
                        "generated_by": "openai",
                    }
            except Exception as e:
                logger.error(f"Erro na geração de resposta OpenAI: {e}")

        response_text = self.response_templates.get(
            category, self.response_templates["improdutivo"]
        )

        return {
            "response": response_text,
            "suggested_actions": self._suggest_actions(category),
            "generated_by": "template",
        }

    def analyze_email(self, email_text: str) -> Dict[str, Any]:
        """
        Análise completa do email: classificação + resposta (processa uma única vez)
        """
        if len(email_text.strip()) < config.MIN_TEXT_LENGTH:
            logger.warning(
                f"Email muito curto para análise: {len(email_text)} caracteres"
            )
            return {
                "category": "improdutivo",
                "confidence": 0.5,
                "method": "validation",
                "reasoning": f"Email muito curto (mínimo {config.MIN_TEXT_LENGTH} caracteres)",
                "response": self.response_templates["improdutivo"],
                "suggested_actions": self._suggest_actions("improdutivo"),
                "generated_by": "template",
            }

        try:
            processed_tokens = preprocess_text(email_text)
            processed_text = " ".join(processed_tokens)
            if not processed_text.strip():
                logger.warning("Texto processado ficou vazio, usando texto original")
                processed_text = email_text.lower().strip()
        except Exception as e:
            logger.error(f"Erro no pré-processamento: {e}")
            processed_text = email_text.lower().strip()

        classification = self._classify_with_processed_text(email_text, processed_text)

        response = self.generate_response(processed_text, classification)

        return {
            "category": classification["category"],
            "confidence": classification["confidence"],
            "method": classification["method"],
            "reasoning": classification.get("reasoning", ""),
            "response": response["response"],
            "suggested_actions": response["suggested_actions"],
            "generated_by": response["generated_by"],
        }

    def _classify_with_processed_text(
        self, email_text: str, processed_text: str
    ) -> Dict[str, Any]:
        """
        Classificação híbrida: combina IA e regras com votação ponderada
        """
        results = []

        # 1. Classificação por regras (sempre disponível)
        rules_result = self._classify_by_rules(email_text)
        results.append(
            {
                "category": rules_result["category"],
                "confidence": rules_result["confidence"],
                "method": "rules",
                "weight": 0.6,  # Peso maior para regras (mais confiável)
                "reasoning": rules_result["reasoning"],
            }
        )

        # 2. Classificação OpenAI (se disponível)
        if self.openai_client.is_available():
            try:
                openai_result = self.openai_client.classify_email(processed_text)
                if (
                    openai_result and openai_result.get("confidence", 0) > 0.3
                ):  # Threshold mais baixo
                    results.append(
                        {
                            "category": openai_result["category"],
                            "confidence": openai_result["confidence"],
                            "method": "openai",
                            "weight": 0.3,  # Peso médio para OpenAI
                            "reasoning": openai_result.get("reasoning", ""),
                        }
                    )
            except Exception as e:
                logger.error(f"Erro na classificação OpenAI: {e}")

        # 3. Classificação Hugging Face (se disponível)
        if self.huggingface_client and self.huggingface_client.is_available():
            try:
                hf_result = self.huggingface_client.classify_email(processed_text)
                if (
                    hf_result and hf_result.get("confidence", 0) > 0.2
                ):  # Threshold mais baixo
                    # Converter score de estrelas (1-5) para confiança (0-1)
                    stars = hf_result.get("confidence", 0)
                    if stars <= 2:
                        hf_category = "improdutivo"
                        hf_confidence = (
                            3 - stars
                        ) / 3  # Inverter: 1 estrela = alta confiança para improdutivo
                    else:
                        hf_category = "produtivo"
                        hf_confidence = (
                            stars - 2
                        ) / 3  # 5 estrelas = alta confiança para produtivo

                    results.append(
                        {
                            "category": hf_category,
                            "confidence": hf_confidence,
                            "method": "huggingface",
                            "weight": 0.1,  # Peso menor para HF
                            "reasoning": hf_result.get("reasoning", ""),
                        }
                    )
            except Exception as e:
                logger.error(f"Erro na classificação Hugging Face: {e}")

        # 4. Votação ponderada
        return self._weighted_vote(results)

    def _weighted_vote(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sistema de votação ponderada para combinar diferentes classificadores
        """
        if not results:
            return {
                "category": "improdutivo",
                "confidence": 0.5,
                "method": "fallback",
                "reasoning": "Nenhum classificador disponível",
            }

        if len(results) == 1:
            return results[0]

        # Calcular scores ponderados
        produtivo_score = 0.0
        improdutivo_score = 0.0
        total_weight = 0.0
        methods_used = []
        reasoning_parts = []

        for result in results:
            weight = result["weight"]
            confidence = result["confidence"]
            category = result["category"]
            method = result["method"]
            reasoning = result.get("reasoning", "")

            # Score ponderado = peso * confiança
            weighted_score = weight * confidence

            if category == "produtivo":
                produtivo_score += weighted_score
            else:
                improdutivo_score += weighted_score

            total_weight += weight
            methods_used.append(method)
            if reasoning:
                reasoning_parts.append(f"{method}: {reasoning}")

        # Normalizar scores
        if total_weight > 0:
            produtivo_score /= total_weight
            improdutivo_score /= total_weight

        # Determinar categoria final
        if produtivo_score > improdutivo_score:
            final_category = "produtivo"
            final_confidence = min(0.95, produtivo_score + 0.1)  # Boost de confiança
        else:
            final_category = "improdutivo"
            final_confidence = min(0.95, improdutivo_score + 0.1)

        # Método híbrido
        methods_str = "+".join(methods_used)

        return {
            "category": final_category,
            "confidence": final_confidence,
            "method": f"hybrid({methods_str})",
            "reasoning": f"Votação ponderada: {produtivo_score:.2f} produtivo vs {improdutivo_score:.2f} improdutivo. "
            + "; ".join(reasoning_parts),
        }

    def _suggest_actions(self, category: str) -> List[str]:
        """Sugere ações baseadas na categoria"""
        if category == "produtivo":
            return [
                "Registrar no sistema de tickets",
                "Verificar status da solicitação",
                "Processar conforme procedimento",
                "Arquivar na pasta do cliente",
                "Responder em até 24h úteis",
            ]
        elif category == "improdutivo":
            return [
                "Arquivar como comunicação social",
                "Não requer ação específica",
                "Manter para referência futura",
            ]
        else:
            return [
                "Avaliar necessidade de resposta",
                "Arquivar para análise posterior",
            ]
