# Agente Inteligente RAG para Cursos Coursera

Este projeto implementa um sistema de **Retrieval-Augmented Generation (RAG)** para recomendação de cursos, desenvolvido para a UC de Introdução à IA.

## Como Executar
1. Instale as bibliotecas: `pip install -r requirements.txt`
2. Garanta que o **Ollama** está a correr com o modelo **Llama 3**.
3. Execute `python indexar.py` para gerar a base de dados vetorial.
4. Inicie o chatbot com `python chatbot.py`.

## Tecnologias
- **LLM:** Llama 3 (via Ollama)
- **Vector Store:** ChromaDB
- **Framework:** LangChain
