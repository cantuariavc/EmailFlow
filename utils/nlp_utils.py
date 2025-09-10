import os
from werkzeug.datastructures import FileStorage
from io import BytesIO
import pdfplumber


ALLOWED_EXTENSIONS = {".txt", ".pdf"}


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
