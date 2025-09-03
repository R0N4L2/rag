"""
evaluate.py - Versión corregida para evitar errores de KV cache y mejorar la robustez.
"""

import os
import datetime
import warnings
from typing import List, Dict, Any
import gc
from datasets import Dataset
from dotenv import load_dotenv
import json

# ============================================
# Configuración de entorno
# ============================================
load_dotenv()
os.environ["RAGAS_EMBEDDINGS"] = "huggingface"
os.environ['RAGAS_DO_NOT_TRACK'] = 'true'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['OPENAI_API_KEY'] = 'DUMMY_KEY_FOR_RAGAS' # Ragas lo necesita aunque no se use

print("[INFO] Variables de entorno configuradas para ejecución local.")

# ============================================
# Importaciones
# ============================================
try:
    from ragas.integrations.langchain import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextRecall,
        ContextPrecision
    )
    from langchain_openai import ChatOpenAI
    from main import RAGSystem
    RAGAS_AVAILABLE = True
    print("[INFO] Todas las dependencias se importaron correctamente.")
except ImportError as e:
    RAGAS_AVAILABLE = False
    warnings.warn(f"[ERROR] No se pudieron importar dependencias clave: {str(e)}. El script no puede continuar.")

# ============================================
# Configuración Global
# ============================================
EVAL_SAMPLES = int(os.getenv('EVAL_SAMPLES', '2'))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
LOCAL_LLM_URL = os.getenv('LOCAL_LLM_URL')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.1))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 512))  
N_THREADS = int(os.getenv('N_THREADS', 4))
MODEL_NAME = os.getenv('MODEL_NAME', 'mistral-7b-instruct-v0.2')

# ============================================
# Dataset de evaluación
# ============================================
def create_evaluation_dataset() -> List[Dict[str, Any]]:
    eval_path = os.getenv("EVALUATION_INPUT_PATH", "evaluate/faq.json")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"No se encontró el archivo de evaluación en {eval_path}")

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Dataset cargado desde {eval_path} con {len(data)} ejemplos.")
    return data

def prepare_ragas_dataset(rag_system: RAGSystem, evaluation_data: List[Dict]) -> Dataset:
    results = {'question': [], 'contexts': [], 'answer': [], 'ground_truth': []}
    total_questions = len(evaluation_data)
    
    for i, item in enumerate(evaluation_data, 1):
        q = item["question"]
        print(f"[INFO] Procesando pregunta {i}/{total_questions}: {q}")
        try:
            response = rag_system.generate_response(q)
            results['question'].append(q)
            # Ragas espera una lista de strings para 'contexts'
            results['contexts'].append([chunk['text'] for chunk in response.get('retrieved_chunks', [])])
            results['answer'].append(response.get('answer', ''))
            results['ground_truth'].append(item["ground_truth"])
        except Exception as e:
            print(f"[ERROR] Falló la pregunta '{q}': {str(e)}")
            # Añadir placeholders para mantener la consistencia del dataset
            results['question'].append(q)
            results['contexts'].append([])
            results['answer'].append("[ERROR DURING GENERATION]")
            results['ground_truth'].append(item["ground_truth"])
            
    print(f"[INFO] Dataset preparado con {len(results['question'])} ejemplos.")

    # Imprimir el contenido del dataset para depuración
    print("\n" + "="*20 + " CONTENIDO DEL DATASET DE EVALUACIÓN " + "="*20)
    dataset = Dataset.from_dict(results)
    for i, row in enumerate(dataset):
        print(f"\n----------- Ejemplo {i+1} -----------")
        print(f"Pregunta: {row['question']}")
        print(f"Respuesta Generada: {row['answer']}")
        # Opcional: imprimir contextos si es necesario (puede ser muy largo)
        # print(f"Contextos: {row['contexts']}")
    print("="*70 + "\n")
    return dataset

# ============================================
# Configuración de modelos locales
# ============================================
# CAMBIO CLAVE: Función factory para crear nuevas instancias del LLM
def create_new_llm_instance() -> ChatOpenAI:
    """Crea y devuelve una nueva instancia del LLM para asegurar un estado limpio."""
    # Forzar la recolección de basura para liberar memoria si es necesario
    gc.collect()
    
    return ChatOpenAI(
        openai_api_base=LOCAL_LLM_URL,
        openai_api_key="not-needed",
        model_name="local-model", # El nombre es irrelevante para servidores locales
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

def setup_local_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Configura y devuelve el modelo de embeddings para Ragas."""
    print(f"[INFO] Cargando modelo de embeddings: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'} # Forzar CPU para embeddings es más estable
    )
    return LangchainEmbeddingsWrapper(embeddings)

# ============================================
# Evaluación RAGAS
# ============================================
def run_ragas_evaluation(dataset: Dataset) -> Dict[str, float]:
    if not RAGAS_AVAILABLE:
        print("[ERROR] RAGAS no está disponible. Abortando evaluación.")
        return {}

    # Configurar embeddings una sola vez
    ragas_embeddings = setup_local_ragas_embeddings()

    # ESTRATEGIA FINAL: Crear una instancia de LLM nueva y limpia para CADA métrica.
    # Esto garantiza un aislamiento total y previene errores de estado del KV cache.
    # Las optimizaciones previas (n_batch, n_threads) estabilizan la creación de instancias.
    metrics = [
        Faithfulness(llm=LangchainLLMWrapper(create_new_llm_instance())),
        AnswerRelevancy(llm=LangchainLLMWrapper(create_new_llm_instance()), embeddings=ragas_embeddings),
        ContextRecall(llm=LangchainLLMWrapper(create_new_llm_instance())),
        ContextPrecision(llm=LangchainLLMWrapper(create_new_llm_instance()))
    ]

    print("[INFO] Ejecutando evaluación RAGAS con instancias de LLM aisladas...")
    result = ragas_evaluate(dataset=dataset, metrics=metrics, raise_exceptions=False)
    print("[INFO] Evaluación RAGAS finalizada.")

    # Convertir el resultado a un diccionario estándar
    scores = result.scores.to_dict()
    metrics_dict = {
        'faithfulness': scores.get('faithfulness', [0.0])[0],
        'answer_relevancy': scores.get('answer_relevancy', [0.0])[0],
        'context_recall': scores.get('context_recall', [0.0])[0],
        'context_precision': scores.get('context_precision', [0.0])[0]
    }
    
    print("[RESULTADOS] Métricas obtenidas:", metrics_dict)
    return metrics_dict

# ============================================
# Función Principal
# ============================================
def main():
    if not RAGAS_AVAILABLE:
        print("[FATAL] Faltan dependencias para ejecutar la evaluación. Revisa la instalación.")
        return

    pdf_path = os.getenv('PDF_PATH', 'data/PDF-Gen-AI-Challenge.pdf')
    if not os.path.exists(pdf_path):
        print(f"[ERROR] No se encontró el archivo PDF en la ruta especificada: {pdf_path}")
        return

    print("[INFO] Inicializando RAGSystem...")
    rag_system = RAGSystem(pdf_path)
    rag_system.setup_system()

    eval_data = create_evaluation_dataset()
    if EVAL_SAMPLES > 0:
        eval_data = eval_data[:EVAL_SAMPLES]
        print(f"[INFO] Usando un subconjunto de {EVAL_SAMPLES} muestras para la evaluación.")

    dataset = prepare_ragas_dataset(rag_system, eval_data)
    
    results = run_ragas_evaluation(dataset)

    if results:
        report_path = os.getenv('EVALUATION_OUTPUT_PATH', 'evaluate/evaluation_report.txt')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("Resultados de la Evaluación RAGAS\n")
                f.write("="*50 + "\n\n")
                f.write(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Modelo Evaluado: {MODEL_NAME}\n")
                f.write(f"Muestras: {len(eval_data)}\n\n")
                f.write("--- Métricas ---\n")
                for metric_name, score in results.items():
                    f.write(f"- {metric_name.replace('_', ' ').title()}: {score:.4f}\n")
            print(f"[SUCCESS] Reporte de evaluación guardado exitosamente en: {os.path.abspath(report_path)}")
        except IOError as e:
            print(f"[ERROR] No se pudo escribir el archivo de reporte en '{report_path}': {e}")
    else:
        print("[ERROR] No se generaron resultados, por lo que no se guardó ningún reporte.")

if __name__ == "__main__":
    main()