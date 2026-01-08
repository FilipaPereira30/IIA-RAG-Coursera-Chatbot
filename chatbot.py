import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

print("--- A iniciar Sistema RAG ---")

# 1. Carregar os Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Ligar à Base de Dados 
if os.path.exists("./db_coursera"):
    vector_db = Chroma(persist_directory="./db_coursera", embedding_function=embeddings)
    print("--- Base de dados carregada! ---")
else:
    print("ERRO: Pasta 'db_coursera' não encontrada!")

# 3. Configurar o Llama 3
llm = Ollama(model="llama3")

print("\n--- Chatbot pronto! Escrever a pergunta (ou 'sair'). ---")

while True:
    pergunta = input("\nTu: ")
    if pergunta.lower() == 'sair': break
    
    try:
        # 4. Busca manual de documentos (Retrieval)
        docs = vector_db.similarity_search(pergunta, k=3)
        contexto = "\n".join([d.page_content for d in docs])
        
        # 5. Criar o Prompt manualmente (Augmentation)
        prompt_final = f"""
        És um assistente do Coursera. Usa o contexto abaixo para responder.
        Contexto: {contexto}
        Pergunta: {pergunta}
        Resposta:"""
        
        # 6. Gerar resposta (Generation)
        resposta = llm.invoke(prompt_final)
        print(f"\nBot: {resposta}")
        
    except Exception as e:
        print(f"Erro ao processar: {e}")