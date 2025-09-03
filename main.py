"""
Sistema RAG para Control de Calidad del Conocimiento Interno

Este módulo implementa el sistema principal de Recuperación Aumentada por Generación (RAG)
que permite realizar búsquedas semánticas sobre documentos PDF y generar respuestas
contextualizadas utilizando un modelo de lenguaje local.

Autor: Ronald Castillo Capino
Email: ron.h.castillo@gmail.com
"""
import os
import json
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from utils import load_and_process_pdf, create_vector_store, chunk_text_semantically
from langchain_openai import ChatOpenAI
# Cargar variables de entorno desde el archivo .env
load_dotenv()

# ============================================
# Configuración del Modelo de Lenguaje
# ============================================
MODEL_NAME = os.getenv('MODEL_NAME', 'mistral-7b-instruct-v0.2')
MODEL_PATH = os.getenv(
    'LOCAL_MODEL_PATH',
    'C:\\Users\\ronal\\.cache\\lm-studio\\models\\jonahhenry\\mistral-7b-instruct-v0.2.Q4_K_M-GGUF\\' \
    'mistral-7b-instruct-v0.2.Q4_K_M.gguf'
)

# ============================================
# Configuración del Sistema RAG
# ============================================
# Configuración de fragmentación y recuperación
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '4000'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
TOP_K_RETRIEVAL = int(os.getenv('TOP_K_RETRIEVAL', '5'))

# Configuración del LLM
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.1'))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '512'))

# Parámetros de rendimiento para LlamaCpp
N_CTX = int(os.getenv('N_CTX', '2048'))
N_GPU_LAYERS = int(os.getenv('N_GPU_LAYERS', '0'))
N_BATCH = int(os.getenv('N_BATCH', '8'))
N_THREADS = int(os.getenv('N_THREADS', '4'))  # Ajusta a tus núcleos físicos

# Configuración de la evaluación
EVAL_SAMPLES = int(os.getenv('EVAL_SAMPLES', '2'))  # Usar 0 para evaluar todos

# Modelo de embeddings
EMBEDDING_MODEL = os.getenv(
    'EMBEDDING_MODEL',
    'sentence-transformers/all-MiniLM-L6-v2'
)

# Suprimir advertencias de GitPython
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

class RAGSystem:
    def __init__(self, pdf_path: str):
        """
        Inicializa el sistema RAG con la ruta al documento PDF de conocimiento.
        
        Este constructor se encarga de:
        1. Cargar el modelo de embeddings
        2. Procesar el PDF en fragmentos
        3. Crear el almacén vectorial
        4. Inicializar el modelo de lenguaje
        
        Args:
            pdf_path (str): Ruta al archivo PDF que servirá como base de conocimiento.
                           El archivo será procesado y cargado en memoria para búsquedas.
        """
        self.pdf_path = pdf_path
        self.model_path = MODEL_PATH
        self.local_llm_url = os.getenv('LOCAL_LLM_URL')
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.vector_store = None
        self.text_chunks = []
        self.metadata = []
        
        # Plantilla de prompt para el LLM
        self.prompt_template = """
[INST]
You are a highly specialized assistant for the book "An Introduction to Statistical Learning with Applications in Python". Your task is to answer questions strictly based on the provided context. Follow these steps and rules meticulously.

**Step-by-Step Instructions:**
1.  Carefully read the user's question and the entire context provided.
2.  Identify the specific parts of the context that directly answer the question.
3.  Synthesize the information from the relevant context into a concise and precise answer.
4.  For each piece of information you use, you MUST cite the source page number at the end of the sentence, like this: `(p. 45)`. The context provides the page number for each source.
5.  If the context does not contain the information needed to answer the question, you MUST respond with exactly this phrase: "The information is not available in the provided context."

**Crucial Rules:**
- DO NOT use any information outside of the provided context.
- DO NOT make assumptions or infer information not explicitly stated.
- Combine information from multiple sources if necessary, citing each one.

**Example:**

**Context:**
---
[Source: Page 25]
Linear regression is a statistical method for modeling the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship.

[Source: Page 88]
Logistic regression is used for binary classification problems. Unlike linear regression, it models the probability of an outcome.
---

**Question:** What is the difference between linear and logistic regression?

**Answer:**
Linear regression models the relationship between a dependent variable and independent variables, assuming a linear relationship (p. 25). In contrast, logistic regression is used for binary classification and models the probability of an outcome (p. 88).
[/INST]

**Task:**

**Context:**
---
{context}
---

**Question:** {question}

**Answer:**
"""

    def setup_system(self):
        """
        Configura el sistema completo: carga datos, procesa y crea vector store
        """
        print("1. Cargando y procesando PDF...")
        
        # Cargar y procesar el PDF
        raw_text, self.metadata = load_and_process_pdf(self.pdf_path)
        
        print("2. Fragmentando texto de manera semántica...")
        # Fragmentación semántica
        self.text_chunks = chunk_text_semantically(raw_text, self.metadata)
        
        print("3. Cargando modelo de embeddings...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        print("4. Creando vector store con FAISS...")
        self.vector_store = create_vector_store(self.text_chunks, self.embedding_model)
        
        print(f"5. Conectando al servidor LLM en: {self.local_llm_url}")

        # Inicializar el cliente para el modelo local
        self.llm = ChatOpenAI(
            openai_api_base=self.local_llm_url,
            openai_api_key="not-needed",
            model_name=MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        print("\nSistema RAG configurado y listo para recibir consultas.")
        print(f"Total de fragmentos: {len(self.text_chunks)}")

    def retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Recupera los fragmentos más relevantes para una consulta
        
        Args:
            query (str): Consulta del usuario
            k (int): Número de fragmentos a recuperar
            
        Returns:
            List[Dict]: Lista de fragmentos con metadatos
        """
        if self.vector_store is None:
            raise ValueError("Sistema no configurado. Ejecuta setup_system() primero.")
        
        # Generar embedding de la consulta
        query_embedding = self.embedding_model.encode([query])
        
        # Buscar fragmentos similares
        scores, indices = self.vector_store.search(query_embedding.astype('float32'), k)
        
        # Recuperar fragmentos con metadatos
        relevant_chunks = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # FAISS retorna -1 para índices no válidos
                chunk_data = {
                    'text': self.text_chunks[idx]['text'],
                    'metadata': self.text_chunks[idx]['metadata'],
                    'similarity_score': float(score),
                    'rank': i + 1
                }
                relevant_chunks.append(chunk_data)
        
        return relevant_chunks

    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Genera una respuesta completa usando RAG
        
        Args:
            query (str): Pregunta del usuario
            
        Returns:
            Dict: Respuesta con metadatos
        """
        # Recuperar contexto relevante
        relevant_chunks = self.retrieve_context(query)
        
        # Combinar fragmentos en contexto
        context_parts = []
        sources_used = []
        
        for chunk in relevant_chunks:
            page_number = chunk['metadata'].get('page_number', 'N/A')
            context_parts.append(f"[Source: Page {page_number}]\n{chunk['text']}")
            sources_used.append({
                'source': chunk['metadata']['source'],
                'page': chunk['metadata'].get('page_number', 'N/A'),
                'section': chunk['metadata'].get('section_title', 'N/A'),
                'similarity_score': chunk['similarity_score']
            })
        
        context = "\n\n".join(context_parts)
        
        # Formatear prompt
        formatted_prompt = self.prompt_template.format(
            context=context,
            question=query
        )
        
        # Generar respuesta usando el modelo local
        try:
            response = self.llm.invoke(formatted_prompt)
            answer = response.content.strip()
        except Exception as e:
            answer = f"Error al generar la respuesta: {str(e)}"
        
        return {
            'query': query,
            'answer': answer,
            'sources_used': sources_used,
            'context_retrieved': context,
            'retrieved_chunks': relevant_chunks
        }

    def interactive_chat(self):
        """
        Bucle interactivo para hacer preguntas al sistema
        """
        print("\nSistema RAG listo. Escribe 'quit' para salir.")
        print("="*60)
        
        while True:
            query = input("\nTu pregunta: ").strip()
            
            if query.lower() in ['quit', 'exit', 'salir']:
                print("¡Hasta luego!")
                break
            
            if not query:
                continue
            
            print("\nProcesando...")
            result = self.generate_response(query)
            
            print(f"\nRespuesta:")
            print("-" * 40)
            print(result['answer'])
            
            print(f"\nFuentes utilizadas:")
            for i, source in enumerate(result['sources_used'], 1):
                print(f"{i}. {source['source']} (Página: {source['page']}, "
                      f"Similitud: {source['similarity_score']:.3f})")

def main():
    """
    Función principal para ejecutar el sistema RAG
    """
    PDF_PATH = "data/PDF-GenAI-Challenge.pdf"
    
    if not os.path.exists(PDF_PATH):
        print(f"Error: No se encontró el archivo {PDF_PATH}")
        return
    
    if not MODEL_PATH:
        print("Error: No se configuró la ruta del modelo local en .env (LOCAL_MODEL_PATH)")
        return
    
    # Inicializar sistema RAG
    rag_system = RAGSystem(PDF_PATH)
    
    # Configurar sistema
    rag_system.setup_system()
    
    # Iniciar chat interactivo
    rag_system.interactive_chat()

if __name__ == "__main__":
    main()