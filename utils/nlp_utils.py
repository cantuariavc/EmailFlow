import os
import logging
from werkzeug.datastructures import FileStorage
from io import BytesIO
import pdfplumber
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from config import Config

logger = logging.getLogger(__name__)
config = Config()

ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
URL_PATTERN = r"https?://\S+|www\.\S+"
PHONE_PATTERN = r"\(\d{2}\)\s*\d{4,5}-?\d{4}"
CPF_PATTERN = r"\b\d{2,3}\.\d{3}\.\d{3}-?\d{2}\b"
PUNCTUATION_PATTERN = r"[^\w\sáàâãéèêíìîóòôõúùûç]"
NUMBERS_PATTERN = r"\b\d+\b"
WHITESPACE_PATTERN = r"\s+"

MIN_TOKEN_LENGTH = config.MIN_TOKEN_LENGTH
MIN_TEXT_LENGTH = config.MIN_TEXT_LENGTH

stemmer = PorterStemmer()

try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    stop_words: set[str] = set(stopwords.words("portuguese"))
except Exception as e:
    logger.warning(f"Erro ao baixar dados NLTK: {e}")
    stop_words: set[str] = {
        "de",
        "da",
        "do",
        "das",
        "dos",
        "para",
        "com",
        "em",
        "na",
        "no",
        "nas",
        "nos",
        "por",
        "pelo",
        "pela",
        "pelos",
        "pelas",
        "a",
        "o",
        "as",
        "os",
        "um",
        "uma",
        "uns",
        "umas",
        "e",
        "ou",
        "mas",
        "se",
        "que",
        "quando",
        "onde",
        "como",
        "porque",
        "então",
        "já",
        "ainda",
        "sempre",
        "nunca",
        "também",
        "aqui",
        "ali",
        "lá",
        "este",
        "esta",
        "estes",
        "estas",
        "esse",
        "essa",
        "esses",
        "essas",
        "aquele",
        "aquela",
        "aqueles",
        "aquelas",
        "meu",
        "minha",
        "meus",
        "minhas",
        "teu",
        "tua",
        "teus",
        "tuas",
        "seu",
        "sua",
        "seus",
        "suas",
        "nosso",
        "nossa",
        "nossos",
        "nossas",
        "deles",
        "delas",
        "neles",
        "nelas",
    }


def extract_text_from_file(file: FileStorage) -> str:
    """
    Lê o conteúdo de um arquivo enviado pelo formulário
    e retorna o texto como string.
    Suporta .txt e .pdf
    """
    email_text = ""

    _, filename_extension = os.path.splitext(str(file.filename))

    if not filename_extension or filename_extension.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Formato de arquivo não suportado. Use .txt ou .pdf")

    if filename_extension == ".pdf":
        try:
            with pdfplumber.open(BytesIO(file.read())) as pdf:
                for page in pdf.pages:
                    email_text += page.extract_text() or " "
        except Exception as e:
            raise ValueError(f"Erro ao ler o PDF: {e}")
    elif filename_extension == ".txt":
        try:
            email_text = file.read().decode("utf-8")
        except UnicodeDecodeError:
            try:
                file.seek(0)
                email_text = file.read().decode("latin-1")
            except UnicodeDecodeError:
                file.seek(0)
                email_text = file.read().decode("utf-8", errors="ignore")

    return email_text


def preprocess_text(email_text: str) -> list[str]:
    """
    Pré-processa texto removendo pontuações, números, e-mails e stop words.
    Também aplica stemming. Mantém palavras importantes para o significado do texto.
    """
    if not email_text or not email_text.strip():
        return []

    if len(email_text.strip()) < MIN_TEXT_LENGTH:
        return []

    email_text = email_text.lower().strip()

    email_text = re.sub(EMAIL_PATTERN, " ", email_text)
    email_text = re.sub(URL_PATTERN, " ", email_text)

    email_text = re.sub(PHONE_PATTERN, " ", email_text)
    email_text = re.sub(CPF_PATTERN, " ", email_text)

    email_text = re.sub(PUNCTUATION_PATTERN, "", email_text)
    email_text = re.sub(NUMBERS_PATTERN, "", email_text)
    email_text = re.sub(WHITESPACE_PATTERN, " ", email_text).strip()

    important_words = {
        "não",
        "sim",
        "só",
        "mais",
        "menos",
        "muito",
        "pouco",
        "bom",
        "boa",
        "bem",
        "mal",
        "melhor",
        "pior",
        "dados",
        "conta",
        "caso",
        "ajuda",
        "como",
        "tudo",
        "valor",
        "dinheiro",
        "pagamento",
        "saldo",
        "extrato",
        "cartão",
        "banco",
        "cliente",
        "serviço",
        "produto",
        "investimento",
        "aplicação",
        "rendimento",
        "taxa",
        "juros",
        "prazo",
        "vencimento",
        "boleto",
        "transferência",
        "depósito",
        "saque",
        "crédito",
        "débito",
        "financiamento",
        "empréstimo",
        "seguro",
        "apólice",
        "prêmio",
        "sinistro",
        "cobertura",
        "beneficiário",
    }

    stop_words_filtered = stop_words - important_words

    try:
        tokens = word_tokenize(email_text)
        processed_tokens: list[str] = []

        for token in tokens:
            if (
                token.strip()
                and len(token) > MIN_TOKEN_LENGTH
                and token not in stop_words_filtered
            ):
                try:
                    stemmed: str = str(stemmer.stem(token))
                    if stemmed and len(stemmed) > MIN_TOKEN_LENGTH:
                        processed_tokens.append(stemmed)
                except Exception:
                    processed_tokens.append(token)

        return processed_tokens

    except Exception as e:
        logger.error(f"Erro no processamento NLTK: {e}")
        words = email_text.split()
        return [
            word
            for word in words
            if word and len(word) > MIN_TOKEN_LENGTH and word not in stop_words_filtered
        ]
