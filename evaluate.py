"""
Sistema RAG para Control de Calidad del Conocimiento Interno
Archivo: evaluate.py
Nombre: Ronald Castillo Capino
Email: ron.h.castillo@gmail.com
Descripción: Evaluación cuantitativa del sistema RAG usando métricas RAGAS
"""

import os
import json
import warnings
import datetime
from typing import List, Dict, Any
from datasets import Dataset
from dotenv import load_dotenv

# Suppress Git warnings
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

# Try to import ragas with suppressed warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy, 
            context_precision,
            context_recall,
            answer_similarity,
            answer_correctness
        )
        from ragas.metrics.critique import harmfulness
        RAGAS_AVAILABLE = True
    except ImportError:
        RAGAS_AVAILABLE = False
        print("Advertencia: No se pudo importar ragas. La evaluación detallada no estará disponible.")

# Configuración del modelo local
LOCAL_LLM_URL = os.getenv('LOCAL_LLM_URL', 'http://localhost:1234/v1')
MODEL_NAME = os.getenv('MODEL_NAME', 'llama-3.1-nemotron-nano-4b-v1.1')

# Configuración de evaluación
EVAL_SAMPLES = int(os.getenv('EVAL_SAMPLES', '0'))  # 0 para evaluar todo
EVALUATION_OUTPUT_PATH = os.getenv('EVALUATION_OUTPUT_PATH', 'evaluate/evaluation_report.txt')

# Asegurar que el directorio de salida exista
os.makedirs(os.path.dirname(EVALUATION_OUTPUT_PATH) or '.', exist_ok=True)

from main import RAGSystem

def create_evaluation_dataset(sample_size: int = 0) -> List[Dict[str, str]]:
    """
    Crea un dataset de evaluación con preguntas y respuestas esperadas
    sobre Statistical Learning
    
    Returns:
        List[Dict]: Dataset de evaluación con preguntas en español y respuestas concisas
    """
    evaluation_data = [
        {
            "question": ["In the context of statistical learning, what is the difference between reducible and irreducible error?"],
            "ground_truth": ["Reducible error is due to an imperfect model and can be decreased by choosing a better algorithm. Irreducible error is due to the inherent variability (noise ε) in the data and represents a lower bound on the test error for any model."]
        },
        {
            "question": ["Why is linear regression not appropriate for classification problems with qualitative responses?"],
            "ground_truth": ["Because encoding a qualitative response with numbers (e.g., 1, 2, 3) imposes an artificial order and distance between classes. Furthermore, the model can produce predictions outside the [0, 1] probability range."]
        },
        {
            "question": ["What is the main difference in assumptions between Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA)?"],
            "ground_truth": ["LDA assumes that all classes share a common covariance matrix, resulting in a linear decision boundary. QDA is more flexible and allows each class to have its own covariance matrix, which results in a quadratic decision boundary."]
        },
        {
            "question": ["In which situation might the Lasso method outperform Ridge regression in predictive accuracy?"],
            "ground_truth": ["Lasso tends to be superior when a relatively small number of predictors have substantial coefficients and the rest are very small or zero. Ridge is better when the response depends on many predictors with similarly sized coefficients."]
        },
        {
            "question": ["What is the advantage of regression splines over polynomial regression for modeling non-linear relationships?"],
            "ground_truth": ["Splines introduce flexibility by adding 'knots' at specific points, allowing the fit to change locally. This avoids the erratic behavior at the boundaries of the data range that is often seen in high-degree polynomials."]
        },
        {
            "question": ["How does a Random Forest improve on the performance of Bagging?"],
            "ground_truth": ["When building each tree, a Random Forest considers only a random subset of predictors at each split. This decorrelates the trees in the ensemble, which reduces the variance of the final prediction compared to Bagging, where all predictors are considered at each split."]
        },
        {
            "question": ["What is the purpose of the 'kernel trick' in Support Vector Machines (SVMs)?"],
            "ground_truth": ["The kernel trick allows operating in a higher-dimensional feature space without explicitly calculating the data's coordinates in that space. This makes it computationally efficient to create non-linear decision boundaries."]
        },
        {
            "question": ["How does Boosting differ from Bagging in tree-based ensemble methods?"],
            "ground_truth": ["In Bagging, trees are built independently on bootstrap samples. In Boosting, trees are built sequentially, where each new tree is fitted to correct the errors (residuals) of the previous trees, thus learning slowly."]
        },
        {
            "question": ["What is 'censoring' in survival analysis and why is it a problem?"],
            "ground_truth": ["Censoring occurs when an individual's survival time is unknown beyond a certain point (e.g., the study ends). It's a problem because these observations cannot be ignored, but their observed time cannot be used as the actual event time, requiring specialized analysis methods."]
        },
        {
            "question": ["What is the difference between controlling the Family-Wise Error Rate (FWER) and the False Discovery Rate (FDR) in multiple hypothesis testing?"],
            "ground_truth": ["Controlling the FWER aims to limit the probability of making at least one Type I error (one false positive). Controlling the FDR is less strict and aims to limit the expected proportion of false discoveries among all rejected hypotheses."]
        },
        {
            "question": ["What role does the non-linear activation function play in a neural network?"],
            "ground_truth": ["The non-linear activation function is essential because without it, a neural network with multiple layers would collapse into a simple linear model. It allows the model to capture complex, non-linear relationships between variables."]
        },
        {
            "question": ["What type of data are Convolutional Neural Networks (CNNs) specialized for?"],
            "ground_truth": ["CNNs are specialized for data with a spatial grid-like structure, such as images. They use convolutional layers to detect local features (edges, textures) which are then combined to form more complex patterns."]
        }
    ]
    
    # Si sample_size > 0, tomar solo una muestra aleatoria
    if sample_size > 0 and sample_size < len(evaluation_data):
        import random
        random.seed(42)  # Para reproducibilidad
        evaluation_data = random.sample(evaluation_data, sample_size)
    
    return evaluation_data

def prepare_ragas_dataset(rag_system: RAGSystem, evaluation_data: List[Dict]) -> Dataset:
    """
    Prepara el dataset para evaluación con RAGAS
    
    Args:
        rag_system: Instancia del sistema RAG
        evaluation_data: Lista de diccionarios con preguntas y respuestas esperadas
        
    Returns:
        Dataset: Dataset formateado para RAGAS
    """
    results = {
        'question': [],
        'contexts': [],
        'answer': [],
        'ground_truth': []
    }
    
    for i, item in enumerate(evaluation_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"\n[{i}/{len(evaluation_data)}] Procesando: {question[:50]}...")
        
        try:
            # Obtener respuesta del sistema RAG
            rag_result = rag_system.generate_response(question)
            answer = rag_result.get('answer', 'No se pudo generar respuesta')
            
            # Obtener contexto recuperado
            context_chunks = rag_system.retrieve_context(question)
            contexts = [chunk['text'] for chunk in context_chunks]
            
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
    Proporciona una interpretación de la métrica y su puntuación
    
    Args:
        metric_name: Nombre de la métrica
        score: Puntuación de la métrica
        
    Returns:
        str: Interpretación de la métrica
    """
    interpretation = {
        'faithfulness': lambda s: f"La respuesta es {'altamente fiel' if s > 0.8 else 'moderadamente fiel' if s > 0.5 else 'poco fiel'} al contexto proporcionado.",
        'answer_relevancy': lambda s: f"La respuesta es {'altamente relevante' if s > 0.8 else 'moderadamente relevante' if s > 0.5 else 'poco relevante'} a la pregunta.",
        'context_precision': lambda s: f"La precisión del contexto es {'excelente' if s > 0.8 else 'buena' if s > 0.5 else 'baja'}.",
        'context_recall': lambda s: f"La recuperación del contexto es {'completa' if s > 0.8 else 'parcial' if s > 0.5 else 'insuficiente'}.",
        'answer_similarity': lambda s: f"La similitud con la respuesta de referencia es {'muy alta' if s > 0.8 else 'moderada' if s > 0.5 else 'baja'}.",
        'answer_correctness': lambda s: f"La corrección de la respuesta es {'alta' if s > 0.8 else 'moderada' if s > 0.5 else 'baja'}.",
        'harmfulness': lambda s: f"El contenido es {'seguro' if s < 0.2 else 'potencialmente problemático' if s < 0.5 else 'peligroso'}."
    }
    
    return interpretation.get(metric_name, lambda _: "Sin interpretación disponible.")(score)


def run_ragas_evaluation(dataset: Dataset) -> Dict[str, Dict[str, Any]]:
    """
    Ejecuta la evaluación RAGAS en el dataset proporcionado
    
    Args:
        dataset: Dataset preparado para evaluación
        
    Returns:
        Dict: Resultados de las métricas con información detallada
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("RAGAS no está disponible para la evaluación")
    
    print("\nIniciando evaluación RAGAS...")
    
    # Definir métricas a evaluar con sus descripciones detalladas
    metrics = [
        ("faithfulness", 
         "Fidelidad (Faithfulness)",
         "Mide cuán fiel es la respuesta generada al contexto proporcionado. Evalúa si la respuesta se puede inferir directamente del contexto dado.",
         "Esta métrica es crucial para garantizar que el sistema no genere información inventada o no respaldada por el contexto.")
    ]
    
    try:
        # Mapeo de nombres de métricas a sus objetos
        metric_map = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall,
            'answer_similarity': answer_similarity,
            'answer_correctness': answer_correctness,
            'harmfulness': harmfulness
        }
        
        # Obtener solo los objetos métrica que están disponibles
        metric_objects = []
        for name, *rest in metrics:
            if name in metric_map:
                metric_objects.append(metric_map[name])
        
        if not metric_objects:
            raise ValueError("No se encontraron métricas válidas para evaluar")
        
        # Ejecutar evaluación
        result = evaluate(
            dataset=dataset,
            metrics=metric_objects,
            raise_exceptions=False
        )
        
        # Procesar resultados con información adicional
        evaluation_results = {}
        for (metric_name, display_name, description, importance), score in zip(metrics, result.values()):
            if metric_name in metric_map:  # Solo procesar métricas válidas
                evaluation_results[metric_name] = {
                    'score': float(score),
                    'display_name': display_name,
                    'description': description,
                    'importance': importance,
                    'interpretation': _interpret_metric(metric_name.split('_')[0], float(score)),
                    'recommendation': get_recommendation(metric_name, float(score))
                }
        
        # Calcular puntuación promedio
        scores = [data['score'] for data in evaluation_results.values() 
                 if isinstance(data, dict) and 'score' in data]
        if scores:
            evaluation_results['average_score'] = {
                'score': sum(scores) / len(scores),
                'description': 'Puntuación promedio de todas las métricas',
                'interpretation': 'Puntuación general del sistema RAG'
            }
        
        return evaluation_results
    
    except Exception as e:
        print(f"[ERROR] Error al ejecutar la evaluación RAGAS: {str(e)}")
        return {}
    
    finally:
        print("\nEvaluación RAGAS finalizada.")


def get_recommendation(metric: str, score: float) -> str:
    """
    Devuelve recomendaciones basadas en la métrica y puntuación
    """
    recommendations = {
        'faithfulness': [
            (0.0, 0.5, "Las respuestas no son consistentes con el contexto. Revisa el prompt y el modelo."),
            (0.5, 0.7, "Mejora la fidelidad ajustando el prompt para que se ciña más al contexto.")
        ],
        'answer_relevancy': [
            (0.0, 0.6, "Las respuestas no son suficientemente relevantes. Mejora el sistema de recuperación."),
            (0.6, 0.8, "Considera refinar el prompt para obtener respuestas más directas a las preguntas.")
        ],
        'context_precision': [
            (0.0, 0.6, "Los fragmentos recuperados no son suficientemente relevantes. Mejora la recuperación."),
            (0.6, 0.8, "Considera ajustar los parámetros de recuperación para obtener contexto más relevante.")
        ],
        'context_recall': [
            (0.0, 0.5, "Se está perdiendo información relevante. Aumenta el número de fragmentos recuperados."),
            (0.5, 0.7, "Considera mejorar la estrategia de recuperación para capturar más contexto relevante.")
        ],
        'answer_similarity': [
            (0.0, 0.6, "Las respuestas generadas difieren significativamente de las esperadas. Revisa el modelo."),
            (0.6, 0.8, "Consideras ajustar el prompt para obtener respuestas más cercanas a las esperadas.")
        ],
        'answer_correctness': [
            (0.0, 0.6, "Las respuestas contienen información incorrecta. Revisa el modelo y los datos de entrenamiento."),
            (0.6, 0.8, "Mejora la precisión de las respuestas ajustando el prompt o el modelo.")
        ]
    }
    
    # Obtener recomendación específica para la métrica
    if metric in recommendations:
        for min_score, max_score, recommendation in recommendations[metric]:
            if min_score <= score < max_score:
                return recommendation
    
    # Recomendación genérica si no hay una específica
    return "Revisa la configuración del sistema para mejorar esta métrica."

    report += "-"*40 + "\n"
    report += f"Modelo: {MODEL_NAME}\n"
    report += f"URL del servidor: {LOCAL_LLM_URL}\n"
    if results:
        report += f"Métricas evaluadas: {', '.join(results.keys())}\n"
    report += "\n" + "="*80 + "\n"
    report += "RESULTADOS DETALLADOS\n"
    report += "="*80 + "\n\n"
    
    # Resultados detallados
    if not results:
        report += "No hay resultados para mostrar.\n"
    else:
        # Métricas principales
        report += "MÉTRICAS PRINCIPALES:\n"
        report += "-"*40 + "\n"
        
        for metric, score in results.items():
            if metric == 'basic_accuracy':
                formatted_score = f"{score*100:.1f}%"
            else:
                formatted_score = f"{score:.4f}"
            report += f"{metric.upper()}: {formatted_score}\n"
        
        # Puntuación promedio (excluyendo basic_accuracy si existe)
        numeric_scores = [v for k, v in results.items() if k != 'basic_accuracy']
        if numeric_scores:
            avg_score = sum(numeric_scores) / len(numeric_scores)
            report += f"\nPUNTAJE PROMEDIO: {avg_score:.4f}\n"
    

def create_detailed_report(rag_system: RAGSystem, results: Dict) -> str:
    """
    Crea un reporte detallado de la evaluación
    
    Args:
        rag_system: Instancia del sistema RAG
        results: Resultados de la evaluación
        
    Returns:
        str: Reporte detallado en formato de texto
    """
    report = []
    report.append("="*80)
    report.append("REPORTE DETALLADO DE EVALUACIÓN DEL SISTEMA RAG".center(80))
    report.append("="*80)
    report.append(f"\nFecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Obtener información del modelo de embedding de manera segura
    model_info = 'Desconocido'
    if hasattr(rag_system, 'embedding_model'):
        if hasattr(rag_system.embedding_model, '_modules') and '0' in rag_system.embedding_model._modules:
            # Para modelos SentenceTransformer
            model_info = rag_system.embedding_model._modules['0'].__class__.__name__
        elif hasattr(rag_system.embedding_model, '__class__'):
            # Para otros tipos de modelos
            model_info = rag_system.embedding_model.__class__.__name__
    
    report.append(f"Modelo de embedding: {model_info}")
    report.append(f"Métricas evaluadas: {', '.join([m for m in results.keys() if m != 'average_score'])}")
    
    # Agregar resumen ejecutivo
    report.append("\n" + "RESUMEN EJECUTIVO".center(80, '-'))
    
    if 'average_score' in results:
        avg = results['average_score']
        report.append(f"\nPuntuación promedio: {avg.get('score', 0):.2f}")
        report.append(f"Interpretación: {avg.get('interpretation', 'No disponible')}")
    
    # Agregar métricas detalladas
    report.append("\n" + "MÉTRICAS DETALLADAS".center(80, '-'))
    
    for metric_name, metric_data in results.items():
        if metric_name != 'average_score' and isinstance(metric_data, dict):
            report.append(f"\n{metric_data.get('display_name', metric_name).upper()}:")
            # Formatear la puntuación de manera segura
            score = metric_data.get('score', 'N/A')
            if isinstance(score, (int, float)):
                score_str = f"{score:.2f}"
            else:
                score_str = str(score)
            report.append(f"  - Puntuación: {score_str}")
            report.append(f"  - Descripción: {metric_data.get('description', 'Sin descripción')}")
            report.append(f"  - Interpretación: {metric_data.get('interpretation', 'Sin interpretación')}")
            
            # Agregar recomendaciones si están disponibles
            if 'recommendation' in metric_data and metric_data['recommendation']:
                report.append("  - Recomendaciones:")
                for rec in metric_data['recommendation'].split('\n'):
                    report.append(f"    * {rec}")
    
    # Agregar conclusión general
    report.append("\n" + "CONCLUSIÓN".center(80, '-'))
    if 'average_score' in results:
        avg_score = results['average_score'].get('score', 0)
        if avg_score >= 0.8:
            conclusion = "El sistema RAG tiene un excelente rendimiento general."
        elif avg_score >= 0.6:
            conclusion = "El sistema RAG tiene un buen rendimiento, pero hay áreas de mejora."
        else:
            conclusion = "El sistema RAG necesita mejoras significativas en su rendimiento."
        report.append(f"\n{conclusion}")
    
    # Convertir a string
    detailed_report = '\n'.join(report)
    
    # Guardar en archivo
    try:
        with open(EVALUATION_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        print(f"[OK] Reporte detallado guardado en: {os.path.abspath(EVALUATION_OUTPUT_PATH)}")
    except Exception as e:
        print(f"[ERROR] Error al guardar el reporte: {str(e)}")
    
    return detailed_report


def display_evaluation_results(results: Dict) -> None:
    """
    Muestra los resultados de la evaluación en la consola
    
    Args:
        results: Diccionario con los resultados de la evaluación
    """
    print("\n" + "="*80)
    print("RESULTADOS DE LA EVALUACIÓN".center(80))
    print("="*80)
    
    # Verificar si es una evaluación básica o RAGAS
    if 'basic_evaluation' in results:
        # Mostrar resultados de evaluación básica
        basic = results['basic_evaluation']
        print(f"\nEvaluación Básica - {basic['samples_evaluated']} muestras evaluadas")
        print("-" * 80)
        
        for i, (key, item) in enumerate(basic['details'].items()):
            print(f"\nPregunta {i+1}:")
            print(f"  - Pregunta: {item['question']}")
            print(f"  - Respuesta esperada: {item['expected_answer']}")
            print(f"  - Respuesta generada: {item['actual_answer']}")
            print(f"  - Evaluación: {item.get('evaluation', 'No evaluado')}")
    else:
        # Mostrar resultados de RAGAS
        print("\nMétricas de Evaluación RAGAS:")
        print("-" * 80)
        
        # Filtrar métricas que no son promedios
        metrics = {k: v for k, v in results.items() if k != 'average_score' and isinstance(v, dict)}
        
        # Mostrar métricas individuales
        for metric_name, metric_data in metrics.items():
            print(f"\n{metric_data.get('display_name', metric_name).upper()}:")
            print(f"  - Puntuación: {metric_data.get('score', 'N/A'):.2f}")
            print(f"  - Descripción: {metric_data.get('description', 'Sin descripción')}")
            print(f"  - Interpretación: {metric_data.get('interpretation', 'Sin interpretación')}")
            print(f"  - Recomendación: {metric_data.get('recommendation', 'Sin recomendación')}")
        
        # Mostrar puntuación promedio si está disponible
        if 'average_score' in results:
            avg = results['average_score']
            print("\n" + "-"*80)
            print(f"PUNTUACIÓN PROMEDIO: {avg.get('score', 0):.2f}")
            print(f"{avg.get('interpretation', '')}")
    
    print("\n" + "="*80 + "\n")


def perform_basic_evaluation(rag_system: RAGSystem, evaluation_data: List[Dict]) -> Dict:
    """
    Realiza una evaluación básica del sistema RAG sin usar RAGAS
    
    Args:
        rag_system: Instancia del sistema RAG
        evaluation_data: Lista de diccionarios con preguntas y respuestas esperadas
        
    Returns:
        Dict: Resultados de la evaluación básica
    """
    print("Realizando evaluación básica...")
    results = {}
    
    for i, item in enumerate(evaluation_data):
        question = item['question'][0] if isinstance(item['question'], list) else item['question']
        expected_answer = item['ground_truth'][0] if isinstance(item['ground_truth'], list) else item['ground_truth']
        
        try:
            # Obtener respuesta del sistema RAG
            response = rag_system.query(question)
            
            # Aquí podrías agregar más métricas básicas si lo deseas
            results[f'query_{i}'] = {
                'question': question,
                'expected_answer': expected_answer,
                'actual_answer': response,
                'evaluation': 'No se pudo evaluar (RAGAS no disponible)'
            }
            
        except Exception as e:
            print(f"Error al procesar la pregunta {i+1}: {str(e)}")
    
    return {
        'basic_evaluation': {
            'status': 'completed',
            'samples_evaluated': len(evaluation_data),
            'details': results
        }
    }


def run_rag_evaluation(rag_system: RAGSystem, evaluation_data: List[Dict]) -> Dict:
    """
    Ejecuta la evaluación del sistema RAG
    
    Args:
        rag_system: Instancia del sistema RAG
        evaluation_data: Lista de diccionarios con preguntas y respuestas esperadas
        
    Returns:
        Dict: Resultados de la evaluación
    """
    print("Preparando datos para evaluación RAGAS...")
    
    try:
        # Preparar dataset para RAGAS
        dataset = prepare_ragas_dataset(rag_system, evaluation_data)
        
        # Ejecutar evaluación RAGAS si está disponible
        if RAGAS_AVAILABLE:
            print("Ejecutando evaluación RAGAS...")
            results = run_ragas_evaluation(dataset)
        else:
            print("RAGAS no está disponible. Realizando evaluación básica...")
            results = perform_basic_evaluation(rag_system, evaluation_data)
            
        return results
        
    except Exception as e:
        print(f"Error durante la evaluación: {str(e)}")
        return {}


def main():
    """
    Función principal para ejecutar la evaluación
    """
    print("Iniciando evaluación del sistema RAG...")
    
    # Verificar que el archivo PDF existe
    pdf_path = os.path.join("data", "PDF-GenAI-Challenge.pdf")
    if not os.path.exists(pdf_path):
        print(f"Error: No se encontró el archivo {pdf_path}")
        print("   Asegúrate de que el PDF esté en la carpeta 'data/'")
        return
    
    # Verificar configuración del servidor local
    if not LOCAL_LLM_URL:
        print("Error: No se configuró la URL del servidor local de LLM.")
        print(f"   Asegúrate de que LM Studio esté ejecutando el modelo {MODEL_NAME}")
        print(f"   y que el servidor API esté habilitado en {LOCAL_LLM_URL}")
        return
    
    print(f"\nConfiguración del modelo:")
    print(f"- Modelo: {MODEL_NAME}")
    print(f"- URL del servidor: {LOCAL_LLM_URL}")
    
    # Inicializar sistema RAG
    try:
        rag_system = RAGSystem(pdf_path)
        rag_system.setup_system()
    except Exception as e:
        print(f"Error al inicializar el sistema RAG: {str(e)}")
        return
    
    # Cargar dataset de evaluación
    print("\nCargando dataset de evaluación...")
    evaluation_data = create_evaluation_dataset(EVAL_SAMPLES)
    
    # Ejecutar evaluación
    print(f"\nEvaluando con {len(evaluation_data)} ejemplos...")
    results = run_rag_evaluation(rag_system, evaluation_data)
    
    # Mostrar resultados
    display_evaluation_results(results)
    
    # Guardar reporte detallado
    create_detailed_report(rag_system, results)
    print(f"\nReporte de evaluación guardado en: {os.path.abspath(EVALUATION_OUTPUT_PATH)}")

if __name__ == "__main__":
    main()