import os
from werkzeug.datastructures import FileStorage
from io import BytesIO
import pdfplumber
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


ALLOWED_EXTENSIONS = {".txt", ".pdf"}

stemmer = PorterStemmer()

try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    stop_words: set[str] = set(stopwords.words("portuguese"))
except Exception as e:
    print(f"Erro ao baixar dados NLTK: {e}")
    stop_words: set[str] = set()


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
        email_text = file.read().decode("utf-8", errors="ignore")

    return email_text


def preprocess_text(email_text: str) -> list[str]:
    """
    Pré-processa texto removendo pontuações, números e stop words. Também aplica stemming.
    Mantém a palavra "não" pois é importante para o significado do texto.
    """
    email_text = email_text.lower()

    email_text = re.sub(r"[^\w\s]|\d+", "", email_text)
    email_text = re.sub(r"\s+", " ", email_text).strip()

    # Remove "não" das stop words para preservá-la
    stop_words_filtered = stop_words - {"não"}

    tokens: list[str] = word_tokenize(email_text)
    processed_tokens: list[str] = [
        stemmer.stem(token)
        for token in tokens
        if token.strip() and token not in stop_words_filtered
    ]

    return processed_tokens
