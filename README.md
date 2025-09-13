# 📧 EmailFlow - Classificador Inteligente de E-mails Financeiros

Uma solução de IA que classifica automaticamente e-mails financeiros em **Produtivos** e **Improdutivos**, gerando respostas para melhorar a eficiência operacional.

## 🚀 Funcionalidades

- **Classificação Automática**: Distingue e-mails produtivos de improdutivos
- **Respostas Inteligentes**: Gera respostas personalizadas usando IA
- **Múltiplos Formatos**: Suporta upload de arquivos PDF e TXT
- **Hierarquia Inteligente**: OpenAI → HuggingFace + Regras → Só Regras
- **Interface Web**: Interface intuitiva e responsiva
- **Deploy na Nuvem**: Hospedado no Heroku

## 🛠️ Tecnologias Utilizadas

- **Backend**: Python 3.13, Flask
- **IA/ML**: OpenAI GPT, HuggingFace Transformers, NLTK
- **Processamento**: PDFplumber, NumPy, Pillow
- **Deploy**: Heroku, Gunicorn
- **Frontend**: HTML, CSS, JavaScript

## 📋 Pré-requisitos

- Python 3.13+
- pip (gerenciador de pacotes Python)
- Conta OpenAI (opcional, para melhor classificação)
- Git

## 🚀 Instalação e Execução Local

### 1. Clone o Repositório

```bash
git clone https://github.com/cantuariavc/EmailFlow.git
cd EmailFlow
```

### 2. Crie um Ambiente Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as Dependências

#### Opção A: Instalação Completa (Recomendada para Desenvolvimento)

```bash
pip install -r requirements-full.txt
```

#### Opção B: Instalação Mínima (Apenas Funcionalidades Básicas)

```bash
pip install -r requirements.txt
```

### 4. Configure as Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# OpenAI (opcional - para melhor classificação)
OPENAI_API_KEY=sua_chave_openai_aqui
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=50
OPENAI_TEMPERATURE=0.3

# HuggingFace (opcional - para classificação alternativa)
HUGGINGFACE_ENABLED=True
HUGGINGFACE_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest

# Configurações Gerais
MIN_TEXT_LENGTH=10
MIN_TOKEN_LENGTH=2
```

### 5. Baixe os Dados do NLTK

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 6. Execute a Aplicação

```bash
flask run
```

A aplicação estará disponível em: `http://127.0.0.1:5000`

## 📖 Como Usar

### 1. Acesse a Interface

Abra seu navegador e acesse:
- **Local**: `http://127.0.0.1:5000`
- **Heroku**: `https://emailflow-902deaa27695.herokuapp.com`

### 2. Faça Upload de um Email

- Clique em "Escolher arquivo" ou "Arraste e solte"
- Selecione um arquivo PDF ou TXT
- Clique em "Analisar Email"

### 3. Visualize os Resultados

A aplicação mostrará:
- **Categoria**: Produtivo ou Improdutivo
- **Resposta Sugerida**: Texto pronto para enviar

## ⚙️ Configurações Avançadas

### Hierarquia de Classificação

O sistema usa uma hierarquia inteligente:

1. **OpenAI** (se disponível) - Classificação mais precisa
2. **HuggingFace + Regras** (se OpenAI indisponível) - Combinação inteligente
3. **Só Regras** (fallback) - Padrões baseados em palavras-chave

## 📊 Estrutura do Projeto

```
nlp_preprocessing/
├── app.py                          # Aplicação principal Flask
├── config.py                       # Configurações
├── requirements-full.txt           # Dependências completas
├── requirements.txt                # Dependências essenciais
├── Procfile                        # Configuração Heroku
├── .python-version                 # Versão Python
├── static/                         # Arquivos estáticos
│   ├── style.css
│   └── script.js
├── templates/                      # Templates HTML
│   └── index.html
└── utils/                          # Módulos utilitários
    ├── financial_email_classifier.py
    ├── huggingface_client.py
    ├── nlp_utils.py
    └── openai_client.py
```

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🙏 Agradecimentos

- OpenAI pela API GPT
- HuggingFace pelos modelos de transformadores
- NLTK pela biblioteca de processamento de linguagem natural
- Flask pela framework web
- Heroku pela plataforma de deploy
