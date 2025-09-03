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
from langchain_community.llms import LlamaCpp
# Cargar variables de entorno desde el archivo .env
load_dotenv()

# ============================================
# Configuración del Modelo de Lenguaje
# ============================================
MODEL_NAME = os.getenv('MODEL_NAME', 'llama-3.1-nemotron-nano-8b-v1')
MODEL_PATH = os.getenv(
    'LOCAL_MODEL_PATH',
    'C:\\Users\\ronal\\.cache\\lm-studio\\models\\unsloth\\' \
    'Llama-3.1-Nemotron-Nano-8B-v1-GGUF\\' \
    'Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_S.gguf'
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
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '2048'))

# Parámetros de rendimiento para LlamaCpp
N_CTX = int(os.getenv('N_CTX', '16384'))
N_GPU_LAYERS = int(os.getenv('N_GPU_LAYERS', '30'))
N_BATCH = int(os.getenv('N_BATCH', '512'))
N_THREADS = int(os.getenv('N_THREADS', '6'))  # Ajusta a tus núcleos físicos

# Configuración de la evaluación
EVAL_SAMPLES = int(os.getenv('EVAL_SAMPLES', '5'))  # Usar 0 para evaluar todos

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
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.vector_store = None
        self.text_chunks = []
        self.metadata = []
        
        # Plantilla de prompt para el LLM
        self.prompt_template = """
You are an expert assistant on the book "An Introduction to Statistical Learning with Applications in Python".
Answer the user’s question based ONLY on the context provided below.

IMPORTANT RULES:
1. If the answer is not in the context, reply: "The information is not available in the provided context"
2. Keep your answers concise and precise
3. Always cite sources using the provided metadata
4. Do not make up any information that is not explicitly stated in the context

Context:
---
{context}
---

Question: {question}

Answer:
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
        
        print(f"5. Cargando modelo LLM desde: {self.model_path}")
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el archivo del modelo en: {self.model_path}")

        # Inicializar el modelo local
        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=N_GPU_LAYERS,  # Usar todas las capas en GPU si está disponible
            n_batch=N_BATCH,
            n_ctx=N_CTX,  # Tamaño de contexto fijo
            n_threads=N_THREADS,
            max_tokens=MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            verbose=True  # Mostrar información de carga
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
            context_parts.append(f"[Fuente: {chunk['metadata']['source']}]\n{chunk['text']}")
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
            answer = response.strip()
        except Exception as e:
            answer = f"Error al generar la respuesta: {str(e)}"
        
        return {
            'query': query,
            'answer': answer,
            'sources_used': sources_used,
            'context_retrieved': context
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