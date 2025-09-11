from flask import Flask, render_template, request, jsonify
from utils.nlp_utils import extract_text_from_file, preprocess_text


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/process")
def process_email():
    email_text = ""

    if "email_file" in request.files and request.files["email_file"].filename != "":
        file = request.files["email_file"]

        try:
            email_text = extract_text_from_file(file)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        if not email_text:
            return jsonify({"error": "Não foi possível ler o arquivo fornecido"}), 400

    elif "email_text" in request.form:
        email_text = request.form["email_text"]

    if not email_text:
        return jsonify({"error": "Nenhum texto ou arquivo de e-mail fornecido"}), 400

    processed_tokens = preprocess_text(email_text)

    return jsonify(
        {
            "email_text": email_text,
            "tokens": processed_tokens,
        }
    )
