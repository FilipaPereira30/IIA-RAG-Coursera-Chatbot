import os
import pandas as pd 
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Carregamento do dataset
print("A carregar dataset...")
dataset = load_dataset("azrai99/coursera-course-dataset")
df = pd.DataFrame(dataset['train'])

# Listar as colunas para sabermos os nomes reais
print(f"Colunas encontradas: {df.columns.tolist()}")

# Ajuste dinâmico dos nomes das colunas (Mapeamento)
col_titulo = 'Course Title' if 'Course Title' in df.columns else 'course_title'
col_desc = 'Course Description' if 'Course Description' in df.columns else 'course_description'

if col_titulo in df.columns and col_desc in df.columns:
    df['full_text'] = "Curso: " + df[col_titulo].astype(str) + " | Descrição: " + df[col_desc].astype(str)
else:
    
    print("Aviso: Colunas esperadas não encontradas. A usar colunas padrão.")
    df['full_text'] = df.iloc[:, 0].astype(str) + " " + df.iloc[:, 1].astype(str)

# 2. Pipeline de Processamento (Chunking)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
docs = text_splitter.create_documents(df['full_text'].tolist())

# 3. Criação da Base de Dados (Fase de Indexação) 
print("A gerar embeddings e base de dados ChromaDB... Isto pode demorar.")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    persist_directory="./db_coursera"
)

print("Sucesso! Pasta 'db_coursera' criada e pronta para o Chatbot.")