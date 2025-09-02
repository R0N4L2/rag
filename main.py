"""
Sistema RAG para Control de Calidad del Conocimiento Interno
Archivo: main.py
Nombre: Ronald Castillo Capino
Email: ron.h.castillo@gmail.com
Descripción: Implementación principal del sistema RAG
"""

import os
import json
import os
from typing import List, Dict, Any
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from utils import load_and_process_pdf, create_vector_store, chunk_text_semantically

# Cargar variables de entorno
load_dotenv()

# Configuración del modelo
LOCAL_LLM_URL = os.getenv('LOCAL_LLM_URL', 'http://localhost:1234/v1')
MODEL_NAME = os.getenv('MODEL_NAME', 'llama-3.1-nemotron-nano-4b-v1.1')  # Modelo local a utilizar

# Configuración del sistema
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
TOP_K_RETRIEVAL = int(os.getenv('TOP_K_RETRIEVAL', '5'))
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.1'))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '500'))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

class RAGSystem:
    def __init__(self, pdf_path: str):
        """
        Inicializa el sistema RAG
        
        Args:
            pdf_path (str): Ruta al archivo PDF de la base de conocimiento
        """
        self.pdf_path = pdf_path
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
        print("Cargando y procesando PDF...")
        
        # Cargar y procesar el PDF
        raw_text, self.metadata = load_and_process_pdf(self.pdf_path)
        
        print("Fragmentando texto de manera semántica...")
        # Fragmentación semántica
        self.text_chunks = chunk_text_semantically(raw_text, self.metadata)
        
        print("Creando embeddings y vector store...")
        # Crear vector store
        self.vector_store = create_vector_store(self.text_chunks, self.embedding_model)
        
        print("Sistema RAG configurado exitosamente!")
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
        
        # Inicializar cliente para el modelo local
        client = OpenAI(
            base_url=LOCAL_LLM_URL,
            api_key='not-needed'  # No se necesita API key para servidor local
        )
        
        # Llamar a la API local
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,  # Asegúrate que este modelo exista en tu servidor local
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en Statistical Learning."},
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                top_p=float(os.getenv('TOP_P', '0.9')),
                frequency_penalty=float(os.getenv('FREQUENCY_PENALTY', '0.0')),
                presence_penalty=float(os.getenv('PRESENCE_PENALTY', '0.6')),
                timeout=30  # Timeout en segundos
            )
            
            # Obtener la respuesta del modelo
            answer = response.choices[0].message.content.strip()
            
        except Exception as e:
            answer = f"Error al conectar con el servicio de OpenAI: {str(e)}"
        
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
    # Configuración
    PDF_PATH = "data/PDF-GenAI-Challenge.pdf"
    
    # Verificar que existe el archivo
    if not os.path.exists(PDF_PATH):
        print(f"Error: No se encontró el archivo {PDF_PATH}")
        print("   Asegúrate de que el PDF esté en la carpeta 'data/'")
        return
    
    # Verificar que el servidor local esté configurado
    if not LOCAL_LLM_URL:
        print("Error: No se configuró la URL del servidor local de LLM.")
        print(f"   Asegúrate de que LM Studio esté ejecutando el modelo {MODEL_NAME}")
        print(f"   y que el servidor API esté habilitado en {LOCAL_LLM_URL}")
        return
    
    # Inicializar sistema RAG
    rag_system = RAGSystem(PDF_PATH)
    
    # Configurar sistema
    rag_system.setup_system()
    
    # Iniciar chat interactivo
    rag_system.interactive_chat()

if __name__ == "__main__":
    main()