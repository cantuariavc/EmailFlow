from flask import Flask, render_template, request, jsonify
from config import Config
from utils.nlp_utils import extract_text_from_file
from utils.financial_email_classifier import FinancialEmailClassifier
import os

config = Config()

app = Flask(__name__)

email_classifier = FinancialEmailClassifier()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze_email():
    """
    Análise completa: categoria + resposta automática
    """
    email_text = _extract_email_text()

    if isinstance(email_text, tuple):
        return email_text

    try:
        result = email_classifier.analyze_email(email_text)

        return jsonify(
            {
                "categoria": result["category"].upper(),
                "confianca": f"{result['confidence']:.1%}",
                "resposta_automatica": result["response"],
                "acoes_sugeridas": result["suggested_actions"],
                "metodo_classificacao": result["method"],
                "gerado_por": result["generated_by"],
                "justificativa": result.get("reasoning", ""),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Erro na análise: {str(e)}"}), 500


def _extract_email_text():
    """
    Função auxiliar para extrair texto do email com validação melhorada
    """
    email_text = ""

    if "email_file" in request.files and request.files["email_file"].filename != "":
        file = request.files["email_file"]

        try:
            email_text = extract_text_from_file(file)
        except ValueError as e:
            return jsonify({"error": f"Erro no arquivo: {str(e)}"}), 400
        except Exception as e:
            return (
                jsonify({"error": f"Erro inesperado ao processar arquivo: {str(e)}"}),
                500,
            )

    elif "email_text" in request.form:
        email_text = request.form["email_text"].strip()

    if not email_text:
        return (
            jsonify(
                {"error": "Nenhum texto ou arquivo (.txt, .pdf) de e-mail fornecido"}
            ),
            400,
        )

    return email_text
