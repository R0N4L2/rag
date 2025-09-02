"""
Sistema RAG para Control de Calidad del Conocimiento Interno
Archivo: utils.py
Nombre: Ronald Castillo Capino
Email: ron.h.castillo@gmail.com
Descripción: Funciones utilitarias para procesamiento de documentos y vector store
"""

import re
import PyPDF2
import faiss
import numpy as np
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer

def clean_text(text: str) -> str:
    """
    Limpia el texto extraído del PDF eliminando artefactos de formato
    
    Args:
        text (str): Texto crudo extraído del PDF
        
    Returns:
        str: Texto limpio
    """
    # Eliminar múltiples espacios en blanco
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminar caracteres especiales problemáticos
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
    
    # Eliminar líneas muy cortas (probablemente encabezados/pies de página)
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
    
    return '\n'.join(cleaned_lines)

def extract_text_from_pdf(pdf_path: str) -> Tuple[str, List[Dict]]:
    """
    Extrae texto y metadatos de un archivo PDF
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        
    Returns:
        Tuple[str, List[Dict]]: Texto extraído y lista de metadatos por página
    """
    full_text = ""
    page_metadata = []
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                cleaned_page_text = clean_text(page_text)
                
                if cleaned_page_text.strip():  # Solo agregar páginas con contenido
                    full_text += cleaned_page_text + "\n\n"
                    
                    # Metadatos de la página
                    page_metadata.append({
                        'page_number': page_num + 1,
                        'char_start': len(full_text) - len(cleaned_page_text) - 2,
                        'char_end': len(full_text) - 2,
                        'source': pdf_path
                    })
    
    except Exception as e:
        raise Exception(f"Error al procesar PDF: {str(e)}")
    
    return full_text, page_metadata

def detect_section_boundaries(text: str) -> List[int]:
    """
    Detecta límites de secciones en el texto basándose en patrones comunes
    
    Args:
        text (str): Texto completo del documento
        
    Returns:
        List[int]: Lista de posiciones donde comienzan nuevas secciones
    """
    boundaries = [0]  # Siempre empezar desde el inicio
    
    # Patrones para detectar nuevas secciones
    patterns = [
        r'\n\s*Chapter\s+\d+',           # Capítulos
        r'\n\s*\d+\.\d+\s+[A-Z]',       # Secciones numeradas (ej: 2.1 Introduction)
        r'\n\s*\d+\s+[A-Z][a-z]+',      # Capítulos numerados (ej: 1 Introduction)
        r'\n\s*[A-Z][a-z]+\s+\d+',      # Secciones con nombre (ej: Exercise 1)
        r'\n\s*\d+\.\s+[A-Z]',          # Listas numeradas importantes
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            boundaries.append(match.start())
    
    # Eliminar duplicados y ordenar
    boundaries = sorted(list(set(boundaries)))
    
    return boundaries

def chunk_text_semantically(text: str, page_metadata: List[Dict], 
                          max_chunk_size: int = 1000, 
                          overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Fragmenta el texto de manera semántica manteniendo coherencia contextual
    
    Args:
        text (str): Texto completo del documento
        page_metadata (List[Dict]): Metadatos de páginas
        max_chunk_size (int): Tamaño máximo de cada fragmento
        overlap (int): Solapamiento entre fragmentos
        
    Returns:
        List[Dict]: Lista de fragmentos con sus metadatos
    """
    # Detectar límites de secciones
    section_boundaries = detect_section_boundaries(text)
    
    chunks = []
    
    for i in range(len(section_boundaries)):
        # Determinar inicio y fin de la sección
        start = section_boundaries[i]
        end = section_boundaries[i + 1] if i + 1 < len(section_boundaries) else len(text)
        
        section_text = text[start:end].strip()
        
        # Si la sección es pequeña, usarla completa
        if len(section_text) <= max_chunk_size:
            if section_text:  # Solo agregar si no está vacío
                # Encontrar página correspondiente
                page_info = find_page_for_position(start, page_metadata)
                
                chunks.append({
                    'text': section_text,
                    'metadata': {
                        'source': 'PDF-GenAI-Challenge.pdf',
                        'page_number': page_info['page_number'],
                        'section_title': extract_section_title(section_text),
                        'char_start': start,
                        'char_end': end,
                        'chunk_type': 'semantic_section'
                    }
                })
        else:
            # Fragmentar sección grande manteniendo coherencia
            section_chunks = chunk_large_section(section_text, start, max_chunk_size, overlap)
            
            for chunk_text, chunk_start, chunk_end in section_chunks:
                page_info = find_page_for_position(chunk_start, page_metadata)
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'source': 'PDF-GenAI-Challenge.pdf',
                        'page_number': page_info['page_number'],
                        'section_title': extract_section_title(chunk_text),
                        'char_start': chunk_start,
                        'char_end': chunk_end,
                        'chunk_type': 'semantic_subsection'
                    }
                })
    
    return chunks

def chunk_large_section(text: str, base_start: int, max_size: int, overlap: int) -> List[Tuple[str, int, int]]:
    """
    Fragmenta una sección grande manteniendo coherencia semántica
    """
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    current_start = base_start
    
    for sentence in sentences:
        # Si agregar esta oración excede el tamaño máximo
        if len(current_chunk + sentence) > max_size and current_chunk:
            # Guardar chunk actual
            chunks.append((
                current_chunk.strip(),
                current_start,
                current_start + len(current_chunk)
            ))
            
            # Iniciar nuevo chunk con solapamiento
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
            current_start = current_start + len(current_chunk) - len(overlap_text) - len(sentence) - 1
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Agregar último chunk
    if current_chunk.strip():
        chunks.append((
            current_chunk.strip(),
            current_start,
            current_start + len(current_chunk)
        ))
    
    return chunks

def find_page_for_position(position: int, page_metadata: List[Dict]) -> Dict:
    """
    Encuentra la página correspondiente para una posición en el texto
    """
    for page_info in page_metadata:
        if page_info['char_start'] <= position <= page_info['char_end']:
            return page_info
    
    # Si no se encuentra, retornar la primera página
    return page_metadata[0] if page_metadata else {'page_number': 1}

def extract_section_title(text: str) -> str:
    """
    Extrae el título de sección del texto
    
    Args:
        text (str): Texto del que extraer el título
        
    Returns:
        str: Título extraído o primeras palabras si no se encuentra un título claro
    """
    if not text or not isinstance(text, str):
        return "Sección sin título"
        
    # Buscar patrones de títulos al inicio del texto
    lines = [line for line in text.split('\n') if line.strip()][:3]  # Revisar primeras 3 líneas no vacías
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Patrón de título (empieza con número o es una línea corta en mayúsculas)
        if re.match(r'^\d+\.?\d*\s+[A-Z]', line):
            return line.strip()
            
        # Línea corta que probablemente sea título
        if len(line) < 60 and len(line.split()) <= 8 and line[0].isupper():
            return line.strip()
    
    # Si no se encuentra título, usar las primeras palabras significativas
    words = [w for w in text.split() if w.strip()][:8]
    if not words:
        return "Sección sin título"
        
    title = ' '.join(words)
    return (title + '...') if len(words) == 8 else title

def load_and_process_pdf(pdf_path: str) -> Tuple[str, List[Dict]]:
    """
    Función principal para cargar y procesar el PDF
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        
    Returns:
        Tuple[str, List[Dict]]: Texto procesado y metadatos
    """
    print(f"Extrayendo texto de {pdf_path}...")
    text, metadata = extract_text_from_pdf(pdf_path)
    
    print(f"Texto extraído: {len(text)} caracteres, {len(metadata)} páginas")
    
    return text, metadata

def create_vector_store(text_chunks: List[Dict], embedding_model: SentenceTransformer) -> faiss.IndexFlatIP:
    """
    Crea un vector store FAISS con los fragmentos de texto
    
    Args:
        text_chunks (List[Dict]): Lista de fragmentos con metadatos
        embedding_model: Modelo de embeddings
        
    Returns:
        faiss.IndexFlatIP: Índice FAISS configurado
    """
    # Extraer solo el texto para generar embeddings
    texts = [chunk['text'] for chunk in text_chunks]
    
    print(f"Generando embeddings para {len(texts)} fragmentos...")
    
    # Generar embeddings
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
    # Normalizar embeddings para usar producto interno como similitud del coseno
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Crear índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product (equivale a coseno con vectores normalizados)
    
    # Agregar embeddings al índice
    index.add(embeddings.astype('float32'))
    
    print(f"Vector store creado: {index.ntotal} vectores de dimensión {dimension}")
    
    return index

def save_system_state(rag_system, save_path: str = "rag_system_state.json"):
    """
    Guarda el estado del sistema para reutilización posterior
    """
    state = {
        'chunks_count': len(rag_system.text_chunks),
        'pdf_path': rag_system.pdf_path,
        'system_ready': rag_system.vector_store is not None
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    
    print(f"Estado del sistema guardado en {save_path}")

# Funciones de utilidad adicionales
def format_sources_citation(sources: List[Dict]) -> str:
    """
    Formatea las fuentes para citación
    """
    citations = []
    for source in sources:
        citation = f"(Página {source['page']}, {source['section']})"
        citations.append(citation)
    
    return "; ".join(citations)

def calculate_chunk_statistics(chunks: List[Dict]) -> Dict[str, Any]:
    """
    Calcula estadísticas de los fragmentos para análisis
    """
    chunk_lengths = [len(chunk['text']) for chunk in chunks]
    
    return {
        'total_chunks': len(chunks),
        'avg_chunk_length': np.mean(chunk_lengths),
        'median_chunk_length': np.median(chunk_lengths),
        'min_chunk_length': min(chunk_lengths),
        'max_chunk_length': max(chunk_lengths),
        'total_characters': sum(chunk_lengths)
    }