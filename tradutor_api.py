"""
Módulo de API FastAPI para tradução automática de Inglês para Francês 
utilizando as classes específicas MarianTokenizer e MarianMTModel 
do Hugging Face para compatibilidade total com o modelo Helsinki-NLP.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MarianTokenizer, MarianMTModel

# Inicializo a aplicação FastAPI com parâmetros descritivos
app = FastAPI(
    title="API de Tradução (EN-FR)",
    description="Serviço de tradução automática de inglês para francês com FastAPI e MarianMTModel.",
    version="1.0.3"
)

# Defino o identificador do modelo na Hugging Face Hub
MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"

# Carrego o tokenizador e o modelo específico do Marian para evitar falhas de mapeamento
try:
    tokenizador = MarianTokenizer.from_pretrained(MODEL_NAME)
    modelo = MarianMTModel.from_pretrained(MODEL_NAME)
except Exception as erro:
    raise RuntimeError(f"Erro ao carregar o modelo MarianMT ou tokenizador: {str(erro)}")

# Defino o modelo de dados de entrada via Pydantic para validação do corpo JSON
class RequisicaoTraducao(BaseModel):
    texto: str

@app.post("/traduzir", summary="Traduz um texto do inglês para o francês")
def traduzir_texto(dados: RequisicaoTraducao):
    """
    Recebo um texto em inglês via JSON, realizo a tokenização com Marian, 
    executo a inferência do modelo e devolvo o texto traduzido estruturado em JSON.
    """
    try:
        # Valido se o texto de entrada não está vazio ou preenchido apenas com espaços
        if not dados.texto.strip():
            raise HTTPException(status_code=400, detail="O campo de texto não pode estar vazio.")
        
        # Tokenizo o texto de entrada para o formato aceito pelo modelo Marian
        entradas = tokenizador(dados.texto, return_tensors="pt", padding=True, truncation=True)
        
        # Realizo a geração/tradução dos tokens em francês
        tokens_traduzidos = modelo.generate(**entradas)
        
        # Decodifico os tokens gerados de volta para uma string legível em texto plano
        texto_traduzido = tokenizador.decode(tokens_traduzidos[0], skip_special_tokens=True)
        
        # Retorno a resposta estruturada para o cliente HTTP
        return {
            "texto_original_en": dados.texto,
            "texto_traduzido_fr": texto_traduzido
        }
        
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro interno no processamento da tradução: {str(erro)}")

