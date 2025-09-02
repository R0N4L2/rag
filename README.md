# 📚 Sistema RAG para Control de Calidad del Conocimiento Interno

👨‍💻 **Autor**: Ronald Castillo Capino  
📧 **Contacto**: ron.h.castillo@gmail.com

> 💡 Este proyecto implementa un sistema avanzado de Preguntas y Respuestas (Q&A) que combina recuperación de información con generación de lenguaje natural, garantizando respuestas precisas y trazables.

## 🚀 Descripción del Proyecto

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** que combina:

- 🔍 **Recuperación de información** de documentos internos
- 🧠 **Generación de respuestas** usando modelos de lenguaje
- 📊 **Evaluación automática** de la calidad de las respuestas

💡 **Caso de Uso Principal**: Sistema de preguntas y respuestas sobre el libro "An Introduction to Statistical Learning with Applications in Python"

## 🎯 Objetivo

Desarrollar un asistente de IA que:

✅ Proporcione respuestas precisas basadas en documentos específicos  
✅ Mantenga la trazabilidad de las fuentes de información  
✅ Evalúe automáticamente la calidad de las respuestas  
✅ Sea fácil de implementar y mantener

## Arquitectura del Sistema

```
Query → Embedding → Retrieval (FAISS) → Augmentation → Generation (Local LLM) → Response
```

### Componentes Principales:
- **Procesador de Documentos**: Extracción y limpieza de texto desde PDF
- **Fragmentación Semántica**: División inteligente manteniendo coherencia contextual
- **Vector Store**: FAISS para búsqueda eficiente de similitud
- **Generación Aumentada**: Modelo local `llama-3.1-nemotron-nano-4b-v1.1` ejecutado a través de LM Studio
- **API Local**: Servidor local en `http://localhost:1234` para servir el modelo

## Estructura del Proyecto

```
rag_challenge/
├── data/
│   └── PDF-GenAI-Challenge.pdf
├── main.py                 # Sistema principal RAG
├── utils.py               # Funciones utilitarias
├── evaluate.py            # Evaluación con RAGAS
├── requirements.txt       # Dependencias
├── README.md             # Este archivo
└── .env                  # Variables de entorno (crear)
```

## ⚙️ Instalación

### 1. 🏗️ Configurar entorno virtual
```bash
# Crear entorno
python -m venv rag_env

# Activar entorno (Linux/Mac)
source rag_env/bin/activate

# Activar entorno (Windows)
rag_env\Scripts\activate
```

### 2. 📦 Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. 🔐 Configurar variables de entorno
Crea un archivo `.env` en la raíz del proyecto con:
```
# Configuración del modelo local
LOCAL_LLM_URL=http://localhost:1234/v1
MODEL_NAME=llama-3.1-nemotron-nano-4b-v1.1

# Ruta de salida para reportes de evaluación
EVALUATION_OUTPUT_PATH=evaluate/evaluation_report.txt
```

### 4. 📂 Preparar datos
Coloca tu documento PDF en la carpeta `data/` con el nombre `PDF-GenAI-Challenge.pdf`

## 💻 Uso del Sistema

### 1. Iniciar el sistema principal
```bash
python main.py
```

### 2. Hacer preguntas
El sistema te permitirá hacer preguntas sobre el contenido del libro. Por ejemplo:
- "¿Qué es el trade-off entre sesgo y varianza?"
- "Explícame los conceptos básicos de regresión lineal"
- "¿Cuál es la diferencia entre aprendizaje supervisado y no supervisado?"

### 3. Evaluar el sistema
Para evaluar el rendimiento del sistema:
```bash
python evaluate.py
```

📊 Esto generará un reporte detallado en `evaluate/evaluation_report.txt`

### Ejemplo de uso:
```
Tu pregunta: ¿Qué es el trade-off entre sesgo y varianza?

Respuesta:
El trade-off entre sesgo y varianza es un concepto fundamental...

Fuentes utilizadas:
1. PDF-GenAI-Challenge.pdf (Página: 28, Similitud: 0.892)
```

## 📊 Evaluación del Sistema

El sistema utiliza RAGAS para evaluar automáticamente la calidad de las respuestas:

### Métricas principales:
- **✅ Fidelidad (Faithfulness)**: Mide si la respuesta se basa únicamente en el contexto proporcionado
- **🎯 Relevancia de Respuesta**: Evalúa qué tan bien la respuesta responde a la pregunta
- **🔍 Precisión de Contexto**: Verifica la calidad de los fragmentos recuperados
- **📚 Recall de Contexto**: Comprueba si se recuperó toda la información relevante

### Ejecutar evaluación completa:
```bash
python evaluate.py
```

### Interpretación de resultados:
- **0.8-1.0**: Excelente rendimiento
- **0.6-0.8**: Buen rendimiento
- **0.4-0.6**: Necesita mejoras
- **<0.4**: Requiere atención inmediata

## 🛠️ Características Técnicas

### 🔧 Preprocesamiento de Datos
- **Extracción de texto** con PyPDF2
- **División inteligente** que mantiene el contexto
- **Metadatos** para seguimiento de fuentes

### 🔢 Generación de Embeddings
- **Modelo**: `sentence-transformers/all-MiniLM-L6-v2`
- **Normalización**: Vectores unitarios para similitud coseno
- **Rendimiento**: Optimizado para búsqueda rápida

### 🔍 Recuperación
- **Motor**: FAISS para búsqueda vectorial
- **Estrategia**: Top-5 fragmentos más relevantes
- **Rendimiento**: Respuestas en tiempo real

### 🤖 Generación
- **Modelo Local**: `llama-3.1-nemotron-nano-4b-v1.1`
- **Técnicas Avanzadas**:
  - Prompt engineering
  - Control de contexto
  - Verificación de hechos

## 📈 Resultados Esperados

El sistema está diseñado para superar los siguientes umbrales:

| Métrica | Objetivo | Explicación |
|---------|----------|-------------|
| 🎯 Fidelidad | > 0.90 | Las respuestas se basan estrictamente en el contexto |
| 🎯 Relevancia | > 0.85 | Las respuestas responden directamente a la pregunta |
| 🎯 Precisión | > 0.80 | Los fragmentos recuperados son relevantes |
| 🎯 Recall | > 0.75 | Se recupera la mayoría de la información relevante |

💡 Estos valores pueden variar según la calidad del documento fuente y la complejidad de las preguntas.

## 🚨 Solución de Problemas

### Problemas Comunes y Soluciones:

1. **🔑 Error de Autenticación**
   ```
   [ERROR] Error de autenticación con el modelo
   ```
   🔧 **Solución**: Verifica que LM Studio esté ejecutándose y que la URL en `.env` sea correcta

2. **📄 Archivo PDF no encontrado**
   ```
   [ERROR] No se encontró el archivo PDF
   ```
   🔧 **Solución**: Asegúrate de que el archivo existe en `data/PDF-GenAI-Challenge.pdf`

3. **🐌 Rendimiento lento**
   ```
   [INFO] La generación de respuestas está tardando más de lo esperado
   ```
   🔧 **Solución**:
   - Reduce el número de fragmentos recuperados
   - Usa un modelo más pequeño
   - Verifica el rendimiento de tu hardware

4. **💾 Problemas de memoria**
   ```
   MemoryError: No se puede asignar memoria
   ```
   🔧 **Solución**:
   - Reduce el tamaño del lote de procesamiento
   - Cierra otras aplicaciones que consuman mucha memoria
   - Considera usar un equipo con más RAM

## 🚀 Próximos Pasos

### Mejoras Planificadas:

1. **🎯 Mejora de Precisión**
   - Implementar re-ranking de resultados
   - Añadir verificación cruzada de hechos
   - Mejorar la recuperación de contexto

2. **⚡ Rendimiento**
   - Optimizar el uso de memoria
   - Implementar caché de embeddings
   - Soporte para procesamiento por lotes

3. **🌐 Interfaz de Usuario**
   - Desarrollar interfaz web interactiva
   - Añadir visualización de fuentes
   - Soporte para múltiples formatos de documentos

4. **📈 Escalabilidad**
   - Soporte para múltiples documentos
   - Búsqueda distribuida
   - Indexación incremental

## 📞 Soporte

### ¿Necesitas ayuda?

1. **📋 Verifica los logs** en la consola para mensajes de error detallados
2. **📊 Revisa el reporte** en `evaluate/evaluation_report.txt`
3. **🔍 Comprueba** la configuración en `.env`
4. **📚 Consulta la documentación** de las dependencias

### ¿Sigues teniendo problemas?

- 📧 Envía un correo a [ron.h.castillo@gmail.com](mailto:ron.h.castillo@gmail.com)
- 🔗 Incluye los mensajes de error y los pasos para reproducir el problema

---

✨ **Desarrollado con ❤️ para el Desafío de Ingeniero de IA** ✨

---

### 📚 Recursos Adicionales

- [Documentación de RAGAS](https://github.com/explodinggradients/ragas)
- [Guía de FAISS](https://github.com/facebookresearch/faiss)
- [Documentación de LM Studio](https://lmstudio.ai/docs/)

🎯 **Objetivo del Proyecto**: Crear un sistema de preguntas y respuestas confiable y escalable para documentación técnica.