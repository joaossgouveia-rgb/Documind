import os
from flask import Flask, render_template, request
from pypdf import PdfReader
from google import genai

app = Flask(__name__)

# Configuração do cliente Gemini através da variável de ambiente GEMINI_API_KEY
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def extrair_texto_e_paginas(arquivo):
    """Extrai o texto do ficheiro (.pdf ou .txt) organizado por páginas ou blocos."""
    blocos = []
    nome = arquivo.filename.lower()

    if nome.endswith('.pdf'):
        reader = PdfReader(arquivo)
        for num, pagina in enumerate(reader.pages, start=1):
            texto = pagina.extract_text() or ""
            if texto.strip():
                blocos.append({"origem": f"Página {num}", "texto": texto.strip()})
    elif nome.endswith('.txt'):
        conteudo = arquivo.read().decode('utf-8', errors='ignore')
        for i, paragrafo in enumerate(conteudo.split('\n\n'), start=1):
            if paragrafo.strip():
                blocos.append({"origem": f"Trecho {i}", "texto": paragrafo.strip()})

    return blocos

def recuperar_contexto_relevante(pergunta, blocos, max_blocos=5):
    """Filtra os blocos do documento com base na relevância das palavras-chave."""
    palavras_chave = [p.lower() for p in pergunta.split() if len(p) > 3]
    pontuados = []

    for bloco in blocos:
        score = sum(1 for p in palavras_chave if p in bloco["texto"].lower())
        pontuados.append((score, bloco))

    # Ordena os trechos pelo número de coincidências
    pontuados.sort(key=lambda x: x[0], reverse=True)
    
    # Se nenhuma palavra coincidir, devolve os primeiros blocos
    selecionados = [b for s, b in pontuados[:max_blocos]] if pontuados[0][0] > 0 else blocos[:max_blocos]
    return selecionados

@app.route('/', methods=['GET', 'POST'])
def index():
    resposta_ia = ""
    fontes_usadas = []
    pergunta = ""

    if request.method == 'POST':
        arquivo = request.files.get('arquivo')
        pergunta = request.form.get('pergunta', '').strip()

        if arquivo and pergunta:
            # 1. Extração do texto
            blocos = extrair_texto_e_paginas(arquivo)
            
            # 2. Recuperação de contexto (Retrieval)
            contexto_blocos = recuperar_contexto_relevante(pergunta, blocos)
            
            # Formatação do contexto para o Prompt
            texto_contexto = "\n\n".join([f"[{b['origem']}]: {b['texto']}" for b in contexto_blocos])
            fontes_usadas = contexto_blocos

            # 3. Geração Aumentada por Recuperação (RAG)
            prompt = f"""
Você é o assistente DocuMind. Responda à pergunta do usuário baseando-se EXCLUSIVAMENTE nas fontes fornecidas abaixo.
Se a informação não estiver presente nas fontes, responda: "Não encontrei essa informação no documento fornecido."

--- FONTES DO DOCUMENTO ---
{texto_contexto}

--- PERGUNTA DO USUÁRIO ---
{pergunta}
"""
            try:
                # Chamada com o modelo atualizado gemini-3.8-flash
                response = client.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=prompt
                )
                resposta_ia = response.text
            except Exception as e:
                resposta_ia = f"Erro ao processar requisição com a IA: {str(e)}"

    return render_template(
        'index.html', 
        resposta=resposta_ia, 
        fontes=fontes_usadas, 
        pergunta=pergunta
    )

if __name__ == '__main__':
    app.run(debug=True)