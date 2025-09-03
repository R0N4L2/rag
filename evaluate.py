"""
Sistema de Evaluación para el Control de Calidad del Conocimiento Interno

Este módulo implementa la evaluación cuantitativa del sistema RAG utilizando métricas RAGAS.
Permite medir la calidad de las respuestas generadas por el modelo en términos de:
- Fidelidad (Faithfulness)
- Relevancia (Answer Relevancy)
- Precisión del Contexto (Context Precision)
- Recuperación del Contexto (Context Recall)

Autor: Ronald Castillo Capino
Email: ron.h.castillo@gmail.com
"""
import os


# Agregar al inicio de evaluate.py, después de las importaciones

# ============================================
# Limpiar variables de entorno problemáticas
# ============================================

# Remover cualquier configuración de API externa que pueda interferir
if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']

if 'OPENAI_MODEL_NAME' in os.environ:
    del os.environ['OPENAI_MODEL_NAME']

if 'OPENAI_BASE_URL' in os.environ:
    del os.environ['OPENAI_BASE_URL']

# Configurar explícitamente que no use APIs externas
os.environ['RAGAS_DO_NOT_TRACK'] = 'true'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("Variables de entorno limpiadas para uso completamente local")

# Configuración de GitPython antes de cualquier otra importación
os.environ['GIT_PYTHON_REFRESH'] = '0'  # 0 = no message, 1 = warning, 2 = exception
os.environ['GIT_PYTHON_DISABLE_OPTIONAL_GIT_BINARY_VALIDATION'] = '1'

import datetime
import warnings
from typing import List, Dict, Any
from datasets import Dataset
from dotenv import load_dotenv

# Importaciones para RAGAS y LangChain
from ragas.integrations.langchain import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import run_config

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# ============================================
# Configuración del Sistema de Evaluación
# ============================================
# Configuración de fragmentación de texto
# Tamaño de fragmentos para procesamiento
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '4000'))
# Solapamiento entre fragmentos
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200')) 
# Fragmentos a recuperar por consulta
TOP_K_RETRIEVAL = int(os.getenv('TOP_K_RETRIEVAL', '5'))

# Configuración del modelo de lenguaje
# Control de creatividad (0.0-1.0)
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.1'))
# Longitud máxima de respuestas
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '2048'))
# Tamaño máximo de contexto
CONTEXT_WINDOW = int(os.getenv('CONTEXT_WINDOW', '131072'))

# Configuración de la evaluación
# Número de muestras para evaluación
EVAL_SAMPLES = int(os.getenv('EVAL_SAMPLES', '5'))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 
                            'sentence-transformers/all-MiniLM-L6-v2')

# Clave de API (no necesaria para modelo local)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'not-needed')
RAGAS_MODEL_ID = os.getenv('RAGAS_MODEL_ID', 
                           'unsloth/Llama-3.1-Nemotron-Nano-8B-v1-GGUF/Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_S.gguf')
LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', 
                            'C:\\Users\\ronal\\.cache\\lm-studio\\models\\unsloth\\Llama-3.1-Nemotron-Nano-8B-v1-GGUF\\Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_S.gguf')
N_CTX = int(os.getenv('N_CTX', '16384'))
N_GPU_LAYERS = int(os.getenv('N_GPU_LAYERS', '30'))
N_BATCH = int(os.getenv('N_BATCH', '512'))
N_THREADS = int(os.getenv('N_THREADS', '6'))

# La configuración de GitPython ya se realizó al inicio del archivo

# ============================================
# Manejo de Dependencias Opcionales
# ============================================
# Las siguientes importaciones son necesarias para la evaluación RAGAS
# pero son manejadas de forma condicional para evitar errores
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness, 
        answer_relevancy, 
        context_recall, 
        context_precision,
        answer_similarity,
        answer_correctness,
        AspectCritic  
    )
    # Definir 'harmfulness' usando AspectCritic (la forma correcta y moderna)
    harmfulness = AspectCritic(
        name="harmfulness",
        definition="Whether the generated answer is harmful, unethical,'\
            ' dangerous, or illegal."
    )
    from langchain_community.llms import LlamaCpp
    
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    warnings.warn(f"""
                  Error al importar dependencias de evaluación: {str(e)}. 
                  La evaluación no se ejecutará.""")

# Importar el sistema RAG
try:
    from main import RAGSystem
except ImportError as e:
    print(f"Error al importar RAGSystem: {str(e)}")
    RAGAS_AVAILABLE = False


# ============================================
# Configuración del Sistema
# ============================================
EMBEDDING_MODEL = os.getenv(
    'EMBEDDING_MODEL',
    'sentence-transformers/all-MiniLM-L6-v2'
)

# Configuración de RAGAS
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_MODEL_NAME"] = RAGAS_MODEL_ID

# La configuración de GitPython ya se realizó al inicio del archivo

# Variable global para verificar la disponibilidad de RAGAS
RAGAS_AVAILABLE = False
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy, 
            context_precision,
            context_recall,
            answer_similarity,
            answer_correctness,
            AspectCritic
        )
        harmfulness = AspectCritic(
            name="harmfulness",
            definition="Whether the generated answer is harmful, "\
                "unethical, dangerous, or illegal."
        )
        from langchain_community.llms import LlamaCpp
        RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"""Advertencia: 
          No se pudo importar las dependencias necesarias: {str(e)}""")
    print("""La evaluación detallada no estará disponible. 
          Instala los paquetes con:""")
    print("pip install ragas langchain-community llama-cpp-python")

# Importar el sistema RAG (asegúrate de que el archivo 'main.py' exista)
try:
    from main import RAGSystem
except ImportError:
    print("""Error: 
          No se pudo importar RAGSystem desde main.py. 
          Asegúrate de que el archivo exista.""")
    exit()

# ============================================
# Configuración del Sistema de Evaluación
# ============================================
# Esta sección define los parámetros y configuraciones
# específicas para la evaluación del modelo RAG
# URL ya configurada anteriormente, usando la variable existente
# para mantener consistencia con la configuración previa
MODEL_NAME = os.getenv('MODEL_NAME', 'llama-3.1-nemotron-nano-8b-v1')
EVAL_SAMPLES = int(os.getenv('EVAL_SAMPLES', '0'))
EVALUATION_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.getenv('EVALUATION_OUTPUT_PATH', 
              'evaluate/evaluation_report.txt')
)

# Asegurar que el directorio de salida exista
os.makedirs(os.path.dirname(EVALUATION_OUTPUT_PATH), exist_ok=True)

# --- FUNCIONES AUXILIARES ---

def create_evaluation_dataset(sample_size: int = 0) -> List[Dict[str, List[str]]]:
    """
    Crea un dataset de evaluación con preguntas y respuestas esperadas sobre Statistical Learning.
    
    Args:
        sample_size (int): Número de muestras a tomar. Si es 0, se usan todas.
    
    Returns:
        List[Dict]: Dataset de evaluación.
    """
    evaluation_data = [
        {"question": "What is the main difference between supervised and unsupervised learning?",
        "ground_truth": "In supervised learning, for each observation of the predictor measurements, there is an associated response measurement. Models are fitted to predict the response based on the predictors. In unsupervised learning, predictor measurements are available, but there is no associated response. The goal is to understand the relationships between the variables or among the observations."},

        {"question": "What is overfitting in machine learning?",
        "ground_truth": "Overfitting refers to a phenomenon where a model learns the training data too well, following the noise or errors in the data too closely. This results in a low error on the training data but a high error on previously unseen test data, as the model has not learned the true underlying signal."},

        {"question": "What is the bias-variance trade-off?",
        "ground_truth": "The bias-variance trade-off is a property of statistical learning methods describing how the expected test error rate decomposes into the sum of the model's squared bias, its variance, and the irreducible error's variance[cite: 51]. Generally, more flexible models have lower bias but higher variance, and vice-versa. Decreasing one tends to increase the other, and the challenge is to find a sweet spot that minimizes the total error[cite: 51]."},

        {"question": "Briefly describe the K-Nearest Neighbors (KNN) method.",
        "ground_truth": "K-Nearest Neighbors (KNN) is a non-parametric method that, for a given test observation, identifies the K training points that are closest to it in the predictor space. For regression, the predicted outcome is the average of the responses of these K neighbors. For classification, the most common class among the K neighbors is predicted[cite: 55, 38]."},

        {"question": "What is regularization used for in linear models?",
        "ground_truth": "Regularization, also known as shrinkage, is used to constrain or shrink the estimated coefficients in linear models. This can significantly reduce the model's variance. Two prominent examples are Ridge regression, which uses an L2 penalty to shrink coefficients, and the Lasso, which uses an L1 penalty that can shrink some coefficients to exactly zero, thus performing variable selection[cite: 58]."},

        {"question": "How does the dimensionality of data impact machine learning algorithms?",
        "ground_truth": "High dimensionality, where the number of predictors 'p' is large, presents significant challenges. Data becomes sparse, distances between observations can lose meaning, and the likelihood of overfitting increases as the model's flexibility grows. This phenomenon, known as the 'curse of dimensionality', can lead a model to fit the training data's noise perfectly but fail to generalize to new data[cite: 58]."},

        {"question": "What is the purpose of cross-validation?",
        "ground_truth": "Cross-validation is a resampling method used to estimate the test error of a statistical learning model[cite: 58]. It allows for assessing the model's performance on unseen data without requiring a separate, explicit test set by splitting the training data into training and validation subsets. k-fold cross-validation is a common method where the data is divided into k folds[cite: 58]."},

        {"question": "Explain the difference between 'bagging' and 'boosting' methods.",
        "ground_truth": "Bagging and Boosting are tree-based ensemble methods. Bagging (Bootstrap Aggregating) reduces variance by training multiple decision trees in parallel on different bootstrap samples of the data and averaging their predictions. Boosting, in contrast, builds trees sequentially, where each new tree is fitted to correct the residual errors of the previous one, turning weak learners into a strong one and reducing both bias and variance[cite: 60]."},

        {"question": "What advantages do non-linear methods offer over linear ones, according to the book?",
        "ground_truth": "Non-linear methods, such as decision trees, splines, or generalized additive models, have the advantage of being able to capture more complex, non-linear relationships between predictors and the response. While linear models assume an additive and linear relationship, non-linear methods are more flexible and can produce more accurate models when the true underlying relationship is not linear[cite: 60]."},

        {"question": "What is the importance of a model's interpretability, according to the context of the book?",
        "ground_truth": "Interpretability refers to the ease with which the relationship between predictors and the response can be understood. While less flexible models are often less accurate, they are more interpretable. Interpretability is crucial when the goal is inference—that is, understanding how changes in the predictors affect the response—which is fundamental for decision-making in many fields[cite: 51]."}
    ]
    
    # Si se especifica, toma una muestra aleatoria para la evaluación.
    if 0 < sample_size < len(evaluation_data):
        import random
        random.seed(42)
        return random.sample(evaluation_data, sample_size)
    
    return evaluation_data

def prepare_ragas_dataset(rag_system: RAGSystem, evaluation_data: List[Dict]) -> Dataset:
    """
    Prepara el dataset en el formato requerido por RAGAS.
    
    Args:
        rag_system: Instancia del sistema RAG configurado
        evaluation_data: Lista de diccionarios con preguntas y respuestas esperadas
        
    Returns:
        Dataset: Dataset en formato compatible con RAGAS
    """
    results = {
        'question': [],
        'contexts': [],
        'answer': [],
        'ground_truth': []
    }
    evaluation_data=[evaluation_data[0]]
    for i, item in enumerate(evaluation_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        q=question if len(question)<50 else question[:50]+'...'
        print(f"\n[{i}/{len(evaluation_data)}] Procesando: {q}")
        
        try:
            # Generar respuesta usando el sistema RAG
            response_dict = rag_system.generate_response(question)
            answer = response_dict.get('answer', '')
            
            # Obtener los contextos usados con manejo de errores
            sources = response_dict.get('sources_used', [])
            contexts = []
            for src in sources:
                if isinstance(src, dict):
                    # Intentar diferentes posibles claves para el texto
                    text = src.get('text') or src.get('content') or str(src)
                    if text and isinstance(text, str):
                        contexts.append(text)
                elif isinstance(src, str):
                    contexts.append(src)
            
            # Si no se encontraron contextos, usar una lista vacía
            if not contexts:
                print("[ADVERTENCIA] No se encontraron contextos para la pregunta")
                contexts = [""]
            
            # Agregar al dataset
            results['question'].append(question)
            results['contexts'].append(contexts)
            results['answer'].append(answer)
            results['ground_truth'].append(ground_truth)
            
            print(f"[OK] Respuesta generada")
            
        except Exception as e:
            print(f"[ERROR] Error al procesar pregunta: {str(e)}")
            continue
    
    return Dataset.from_dict(results)

def _interpret_metric(metric_name: str, score: float) -> str:
    """
    Proporciona una interpretación de la métrica basada en la puntuación.
    """
    interpretation = {
        'faithfulness': lambda s: f"La respuesta es {'altamente fiel' if s > 0.8 else 'moderadamente fiel' if s > 0.5 else 'poco fiel'} al contexto.",
        'answer_relevancy': lambda s: f"La respuesta es {'altamente relevante' if s > 0.8 else 'moderadamente relevante' if s > 0.5 else 'poco relevante'} a la pregunta.",
        'context_precision': lambda s: f"La precisión del contexto es {'excelente' if s > 0.8 else 'buena' if s > 0.5 else 'baja'}.",
        'context_recall': lambda s: f"La recuperación del contexto es {'completa' if s > 0.8 else 'parcial' if s > 0.5 else 'insuficiente'}.",
        'answer_similarity': lambda s: f"La similitud con la respuesta de referencia es {'muy alta' if s > 0.8 else 'moderada' if s > 0.5 else 'baja'}.",
        'answer_correctness': lambda s: f"La corrección de la respuesta es {'alta' if s > 0.8 else 'moderada' if s > 0.5 else 'baja'}.",
        'harmfulness': lambda s: f"El contenido es {'seguro' if s < 0.2 else 'potencialmente problemático' if s < 0.5 else 'peligroso' if s < 0.8 else 'muy peligroso'}"
    }
    
    return interpretation.get(metric_name, lambda _: "Sin interpretación disponible.")(score)

def display_evaluation_results(results: Dict[str, float]):
    """Muestra los resultados de la evaluación en la consola."""
    if not results:
        print("No hay resultados para mostrar.")
        return
        
    print("\n" + "="*80)
    print("RESULTADOS DE LA EVALUACIÓN".center(80))
    print("="*80)
    
    # Mostrar cada métrica con su puntuación, interpretación y descripción
    for metric, score in results.items():
        print(f"\n- {metric.replace('_', ' ').title()}: {score:.4f}")
        print(f"   Interpretación: {_interpret_metric(metric, score)}")
        print(f"   Recomendación: {get_recommendation(metric, score)}")
    
    # Mostrar resumen
    avg_score = sum(results.values()) / len(results) if results else 0
    print("\n" + "-"*80)
    print(f"Puntuación promedio: {avg_score:.4f}")
    
    # Mostrar recomendaciones generales
    print("\n" + "-"*80)
    print("RECOMENDACIONES GENERALES".center(80))
    print("-"*80)
    
    # Identificar métricas que necesitan atención
    needs_attention = {m: s for m, s in results.items() if s < 0.5}
    
    if needs_attention:
        print("Se recomienda priorizar la mejora de las siguientes métricas:")
        for metric, score in needs_attention.items():
            print(f"- {metric.replace('_', ' ').title()} ({score:.4f}): {get_recommendation(metric, score)}")
    else:
        print("Todas las métricas están en un nivel aceptable. Se recomienda continuar con el monitoreo regular.")
    
    print("\n" + "="*80)

def create_detailed_report(rag_system: RAGSystem, results: Dict[str, float]):
    """Crea un reporte detallado de la evaluación."""
    if not results:
        print("No hay resultados para generar el reporte.")
        return
        
    report_path = "evaluate/evaluation_report.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Calcular puntuación promedio
    avg_score = sum(results.values()) / len(results) if results else 0
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Encabezado
        f.write("="*80 + "\n")
        f.write("REPORTE DETALLADO DE EVALUACIÓN DEL SISTEMA RAG\n".center(80))
        f.write("="*80 + "\n\n")
        
        # Información del sistema
        f.write(f"Fecha de evaluación: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modelo LLM: {os.path.basename(LOCAL_MODEL_PATH)}\n")
        f.write(f"Archivo PDF: {os.path.basename(rag_system.pdf_path)}\n")
        f.write(f"Puntuación promedio: {avg_score:.4f}\n\n")
        
        # Resumen ejecutivo
        f.write("-"*80 + "\n")
        f.write("RESUMEN EJECUTIVO\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"El sistema RAG ha obtenido una puntuación general de {avg_score:.4f} "
                f"({_interpret_metric('faithfulness', avg_score).replace('La respuesta es ', '').capitalize()}) en la evaluación. ")
        
        if avg_score >= 0.7:
            f.write("El rendimiento general es muy bueno, con la mayoría de las métricas "
                   "mostrando un rendimiento sólido.\n")
        elif avg_score >= 0.5:
            f.write("El rendimiento es aceptable, pero hay áreas que podrían mejorarse. "
                   "Se recomienda revisar las métricas individuales.\n")
        else:
            f.write("El rendimiento es inferior al esperado. Se recomienda una revisión "
                   "detallada de las métricas individuales.\n")
        
        # Métricas detalladas
        f.write("\n" + "-"*80 + "\n")
        f.write("MÉTRICAS DETALLADAS\n")
        f.write("-"*80 + "\n\n")
        
        for metric, score in results.items():
            f.write(f"- {metric.replace('_', ' ').title()}: {score:.4f}\n")
            f.write(f"   Interpretación: {_interpret_metric(metric, score)}\n")
            f.write(f"   Recomendación: {get_recommendation(metric, score)}\n\n")
        
        # Recomendaciones generales
        f.write("-"*80 + "\n")
        f.write("RECOMENDACIONES GENERALES\n")
        f.write("-"*80 + "\n\n")
        
        # Recomendaciones basadas en métricas bajas
        low_metrics = {m: s for m, s in results.items() if s < 0.5}
        if low_metrics:
            f.write("Se recomienda enfocarse en mejorar las siguientes áreas:\n\n")
            for metric, score in low_metrics.items():
                f.write(f"- {metric.replace('_', ' ').title()} ({score:.4f}): {get_recommendation(metric, score)}\n")
        else:
            f.write("Todas las métricas están en un nivel aceptable o superior. "
                   "Se recomienda continuar con el monitoreo regular del sistema.\n")
        
        # Pie de página
        f.write("\n" + "="*80 + "\n")
        f.write("Fin del reporte de evaluación\n".center(80))
        f.write("="*80 + "\n")
    
    print(f"\nReporte detallado guardado en: {os.path.abspath(report_path)}")

def get_recommendation(metric: str, score: float) -> str:
    """
    Devuelve recomendaciones basadas en la métrica y la puntuación.
    """
    recommendations = {
        'faithfulness': [
            (0.0, 0.3, "Las respuestas no son consistentes con el contexto. Revisa el modelo y el prompt."),
            (0.3, 0.6, "Mejora la fidelidad ajustando el prompt para que se ciña más al contexto."),
            (0.6, 0.8, "La fidelidad es buena, pero podría mejorarse con un ajuste fino del modelo."),
            (0.8, 1.0, "Excelente fidelidad. Las respuestas son consistentes con el contexto.")
        ],
        'answer_relevancy': [
            (0.0, 0.4, "Las respuestas no son relevantes. Revisa el sistema de recuperación y el modelo."),
            (0.4, 0.7, "Mejora la relevancia de las respuestas ajustando el prompt y los parámetros de búsqueda."),
            (0.7, 0.9, "Buena relevancia. Considera ajustes menores para mejorar aún más."),
            (0.9, 1.0, "Excelente relevancia. Las respuestas son muy pertinentes a las preguntas.")
        ],
        'context_precision': [
            (0.0, 0.5, "Baja precisión en la recuperación de contexto. Revisa el sistema de recuperación."),
            (0.5, 0.7, "Precisión moderada. Ajusta los parámetros de búsqueda semántica."),
            (0.7, 0.9, "Buena precisión. Los contextos recuperados son generalmente relevantes."),
            (0.9, 1.0, "Excelente precisión. Los contextos recuperados son altamente relevantes.")
        ],
        'context_recall': [
            (0.0, 0.4, "Se está perdiendo mucha información relevante. Aumenta el número de fragmentos."),
            (0.4, 0.7, "Recuperación moderada. Considera mejorar la estrategia de recuperación."),
            (0.7, 0.9, "Buena recuperación. La mayoría de la información relevante se está recuperando."),
            (0.9, 1.0, "Excelente recuperación. Prácticamente toda la información relevante está siendo recuperada.")
        ],
        'answer_similarity': [
            (0.0, 0.5, "Baja similitud con las respuestas de referencia. Revisa el modelo y el prompt."),
            (0.5, 0.7, "Similitud moderada. Ajusta el prompt para obtener respuestas más cercanas a las referencias."),
            (0.7, 0.9, "Buena similitud. Las respuestas son cercanas a las referencias."),
            (0.9, 1.0, "Excelente similitud. Las respuestas son prácticamente idénticas a las referencias.")
        ],
        'answer_correctness': [
            (0.0, 0.5, "Las respuestas contienen información incorrecta. Revisa el modelo y los datos de entrenamiento."),
            (0.5, 0.8, "Precisión moderada. Mejora la corrección ajustando el prompt o el modelo."),
            (0.8, 0.95, "Buena precisión. Las respuestas son generalmente correctas."),
            (0.95, 1.0, "Excelente precisión. Las respuestas son prácticamente perfectas.")
        ],
        'harmfulness': [
            (0.0, 0.1, "Sin contenido dañino detectado."),
            (0.1, 0.3, "Bajo riesgo de contenido dañino."),
            (0.3, 0.6, "Riesgo moderado de contenido potencialmente problemático."),
            (0.6, 1.0, "Alto riesgo de contenido dañino. Se recomienda implementar filtros de seguridad.")
        ]
    }
    
    for min_score, max_score, recommendation in recommendations.get(metric, [(0, 1.0, "Sin recomendaciones específicas disponibles.")]):
        if min_score <= score < max_score:
            return recommendation

# Reemplaza la sección de configuración de RAGAS en evaluate.py

def setup_local_ragas_models():
    """
    Configura correctamente los modelos locales para RAGAS
    """
    try:
        # 1. Configurar el LLM local con LlamaCpp
        llm = LlamaCpp(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=N_GPU_LAYERS,
            n_batch=N_BATCH,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            verbose=False
        )
        
        # 2. Configurar embeddings locales
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )

        # 3. Envolver con los wrappers de RAGAS
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        
        return ragas_llm, ragas_embeddings
        
    except Exception as e:
        print(f"Error configurando modelos locales: {str(e)}")
        return None, None

def run_ragas_evaluation(dataset: Dataset) -> Dict[str, float]:
    """
    Ejecuta la evaluación RAGAS con modelos completamente locales
    """
    if not RAGAS_AVAILABLE:
        print("RAGAS no está disponible. No se puede ejecutar la evaluación.")
        return {}
    
    if not LOCAL_MODEL_PATH or not os.path.exists(LOCAL_MODEL_PATH):
        print(f"Error: No se encontró el modelo en la ruta: {LOCAL_MODEL_PATH}")
        return {}
    
    print(f"\nConfigurando modelos locales para RAGAS...")
    
    # Configurar modelos locales
    ragas_llm, ragas_embeddings = setup_local_ragas_models()
    
    if not ragas_llm or not ragas_embeddings:
        print("Error: No se pudieron configurar los modelos locales")
        return {}
    
    try:
        # Definir las métricas con los modelos locales
        metrics = [
            faithfulness.evolve(llm=ragas_llm),
            answer_relevancy.evolve(llm=ragas_llm, embeddings=ragas_embeddings),
            context_recall.evolve(llm=ragas_llm),
            context_precision.evolve(llm=ragas_llm),
            answer_similarity.evolve(embeddings=ragas_embeddings),
            answer_correctness.evolve(llm=ragas_llm),
            harmfulness.evolve(llm=ragas_llm),
        ]
        
        print("\nIniciando evaluación con las siguientes métricas:")
        for metric in metrics:
            print(f"- {metric.name}")
            
        # Configurar RAGAS para modo secuencial
        run_config.max_workers = 1
        print("\nEjecutando evaluación en modo secuencial para garantizar estabilidad.")

        # Ejecutar la evaluación sin configuraciones externas de API
        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
        )
        
        # Convertir resultados a diccionario
        metrics_dict = {}
        for metric in metrics:
            metric_name = metric.name
            if hasattr(result, metric_name):
                metrics_dict[metric_name] = float(getattr(result, metric_name))
        
        return metrics_dict
        
    except Exception as e:
        print(f"\nError durante la evaluación: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def check_lmstudio_connection() -> bool:
    """Verifica la conexión con el servidor de LM Studio."""
    try:
        import requests
        response = requests.get(LOCAL_LLM_URL.rstrip('/') + '/models')
        return response.status_code == 200
    except Exception:
        return False

def list_available_models() -> list:
    """Intenta listar los modelos disponibles en el servidor de LM Studio."""
    try:
        import requests
        response = requests.get(LOCAL_LLM_URL.rstrip('/') + '/models')
        if response.status_code == 200:
            return response.json().get('data', [])
    except Exception:
        pass
    return []

def main():
    """
    Función principal para ejecutar la evaluación del sistema RAG.
    """
    pdf_path = os.getenv('PDF_PATH', 'data/PDF-GenAI-Challenge.pdf')
    
    # Verificar archivo PDF
    if not os.path.exists(pdf_path):
        print(f"Error: No se encontró el archivo PDF en '{pdf_path}'")
        return
    
    # Verificar ruta del modelo
    if not LOCAL_MODEL_PATH:
        print("Error: No se configuró la ruta del modelo en .env (LOCAL_MODEL_PATH)")
        return
    
    # Verificar dependencias
    if not RAGAS_AVAILABLE:
        print("Error: No se pueden ejecutar las evaluaciones. Faltan dependencias.")
        print("Por favor, instala los paquetes necesarios: pip install ragas langchain-community llama-cpp-python")
        return
    
    try:
        # Inicializar sistema RAG
        print("\nInicializando el sistema RAG...")
        rag_system = RAGSystem(pdf_path)
        rag_system.setup_system()
        
        # Cargar dataset de evaluación
        print("\nCargando dataset de evaluación...")
        evaluation_data = create_evaluation_dataset(EVAL_SAMPLES)
        
        # Ejecutar evaluación si hay datos
        if evaluation_data:
            print(f"\nEvaluando con {len(evaluation_data)} ejemplos...")
            dataset = prepare_ragas_dataset(rag_system, evaluation_data)
            results = run_ragas_evaluation(dataset)
            
            # Mostrar y guardar resultados
            if results:
                display_evaluation_results(results)
                create_detailed_report(rag_system, results)
            else:
                print("No se generaron resultados de evaluación.")
        else:
            print("No hay datos de evaluación disponibles.")
            
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un error durante la evaluación: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()