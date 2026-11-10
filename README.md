FastAPI - Parte 1: Questão 2 (API de Tradução EN-FR)

Serviço de tradução automática de inglês para francês utilizando FastAPI e o modelo local MarianMT (Helsinki-NLP/opus-mt-en-fr) da Hugging Face, com tratamento nativo para compatibilidade com a versão 5 da biblioteca transformers.
Tecnologias Utilizadas

    Python

    FastAPI / Pydantic

    Hugging Face Transformers (MarianMTModel, MarianTokenizer)

    PyTorch

Como Executar

    Ative o ambiente virtual executando no PowerShell:
    .\venv\Scripts\Activate

    Instale as dependências necessárias executando:
    pip install fastapi uvicorn transformers torch sentencepiece pydantic

    Inicie a aplicação com o Uvicorn executando:
    uvicorn tradutor_api:app --reload

    Acesse o Swagger UI para testes em:
    http://127.0.0.1:8000/docs
