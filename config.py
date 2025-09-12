import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """
    Configurações centralizadas da aplicação.

    Todas as configurações são carregadas de variáveis de ambiente
    com valores padrão apropriados.
    """

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "50"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

    ALLOWED_EXTENSIONS: set[str] = {".txt", ".pdf"}
    MIN_TEXT_LENGTH: int = int(os.getenv("MIN_TEXT_LENGTH", "10"))
    MIN_TOKEN_LENGTH: int = int(os.getenv("MIN_TOKEN_LENGTH", "2"))
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
