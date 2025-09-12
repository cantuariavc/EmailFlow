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

    ALLOWED_EXTENSIONS: set[str] = {".txt", ".pdf"}
    MIN_TEXT_LENGTH: int = int(os.getenv("MIN_TEXT_LENGTH", "10"))
    MIN_TOKEN_LENGTH: int = int(os.getenv("MIN_TOKEN_LENGTH", "2"))
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
