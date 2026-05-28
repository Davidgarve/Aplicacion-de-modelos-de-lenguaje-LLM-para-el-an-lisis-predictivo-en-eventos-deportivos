# Aplicación de modelos de lenguaje LLM para el análisis predictivo en eventos deportivos

Este repositorio aloja el código fuente, los cuadernos de experimentación y la metodología de preprocesamiento desarrollados para el Trabajo de Fin de Grado (TFG) centrado en la anticipación táctica deportiva. El proyecto investiga la viabilidad de emplear Modelos de Visión-Lenguaje (VLMs) y Modelos de Lenguaje Grande (LLMs) ejecutados en entornos locales para inferir la dirección de un lanzamiento de penalti, basándose exclusivamente en la cinemática corporal del lanzador en los instantes previos y simultáneos al golpeo del balón.

## Contexto y Motivación del Proyecto

El análisis anticipatorio en el fútbol, y en particular en los lanzamientos de penalti, se ha abordado tradicionalmente mediante modelos de estimación de pose y redes convolucionales entrenadas desde cero. Sin embargo, estos enfoques a menudo sufren ante la baja resolución, el desenfoque por movimiento (motion blur) y las oclusiones inherentes a las retransmisiones televisivas estándar. 

Este proyecto propone un cambio de paradigma: utilizar modelos fundacionales multimodales (como la familia Qwen-VL) no como clasificadores finales, sino como extractores de características biomecánicas en lenguaje natural. La premisa es evaluar si estas redes neuronales pueden razonar sobre la geometría del cuerpo humano y generar una representación intermedia (memoria visual) lo suficientemente robusta como para que un algoritmo secundario (un LLM o un modelo estadístico clásico) deduzca la lateralidad del disparo.

## Arquitectura General del Sistema

El flujo de trabajo diseñado es estrictamente modular y actúa de forma anticipatoria, sin observar la trayectoria de vuelo del balón. El conducto de inferencia se divide en las siguientes etapas:

1. **Segmentación Temporal:** El vídeo original se normaliza y se divide temporalmente para capturar la secuencia cinemática completa: carrera inicial, planta o apoyo, impacto y seguimiento.
2. **Generación de Memoria Visual:** Un modelo multimodal (VLM) analiza los fotogramas clave y estructura la evidencia visual en un objeto JSON, detallando la orientación de la cadera, el pie de apoyo, la inercia del torso y el seguimiento de la pierna de golpeo.
3. **Inferencia Táctica:** La memoria visual es ingerida por un segundo modelo (ya sea un LLM mediante prompting analítico o un clasificador tradicional como Random Forest), que traduce las señales anatómicas relativas al jugador en coordenadas absolutas de la portería.

## Estructura Detallada del Repositorio

El diseño experimental se ha orquestado a través de cinco cuadernos Jupyter (Notebooks), concebidos para ejecutarse secuencialmente. Cada iteración documenta un pivote metodológico frente a las limitaciones descubiertas en la fase anterior.

### 1_preprocesamiento.ipynb
Actúa como la base estructural del tratamiento de datos. Realiza dos operaciones críticas sobre el dataset original:
* **Reducción de Dimensionalidad Espacial:** Mapea la compleja cuadrícula de 9 zonas de la portería hacia un sistema de decisión simplificado de 3 columnas (Izquierda, Centro, Derecha), eliminando la variable de altura para focalizar el problema en la lateralidad biomecánica.
* **Estandarización Temporal (64 frames):** Implementa un algoritmo heurístico de Padding (relleno estático) y Cut (recorte dinámico) para garantizar que todas las secuencias de vídeo introducidas a la red neuronal posean exactamente 64 fotogramas de duración. La lógica de recorte asegura que la fase del impacto y el seguimiento post-golpeo (el clímax cinemático) se preserve siempre intacta.

### 2_iteracion1_2.ipynb
Constituye la fase de exploración sobre el nivel de granularidad anatómica que los modelos son capaces de interpretar.
* **Iteración 1 (Enfoque Micro-biomecánico):** Intenta extraer microdetalles articulares finos (ángulos de flexión, rotaciones exactas de tobillo) utilizando tanto formulaciones abiertas como catálogos cerrados de etiquetas. Los resultados demuestran la fragilidad de este enfoque ante la compresión de vídeo.
* **Iteración 2 (Enfoque Macro-gestual):** Responde a las limitaciones previas abandonando la micro-cinemática. Redirige la atención del modelo hacia la captura de configuraciones corporales de mayor escala, evaluando la inercia del cuerpo, la inclinación del eje del torso y la trayectoria residual de la pierna de golpeo.

### 3_iteracion3.ipynb
Aborda el problema de las alucinaciones espaciales del modelo de visión introduciendo una capa analítica tradicional.
* **Anclaje Espacial con YOLO-Pose:** Procesa todos los fotogramas mediante YOLOv8-Pose para superponer un esqueleto bidimensional sobre el jugador. Esta asistencia geométrica reduce drásticamente el ruido visual generado por la indumentaria o el fondo, permitiendo al VLM centrarse en interpretar los cruces articulares y la apertura de caderas sin perder el contexto visual del campo.

### 4_iteracion4_5.ipynb
Somete el pipeline a pruebas de estrés actualizando el ecosistema a modelos de mayor capacidad de razonamiento (Qwen-3.5-VL 8B y Llama 3.1 8B).
* **Iteración 4:** Retorno de control al enfoque micro-biomecánico. Se implementa un estudio de ablación de varianza dinámico para identificar y podar etiquetas cerradas redundantes o de escaso valor discriminativo.
* **Iteración 5:** Reevaluación del enfoque macro-gestual con prompts altamente restrictivos. Se disocia la evaluación del motor de visión frente al motor textual, comparando las conclusiones de Llama frente a Qwen para certificar si un error predictivo emana de una "ceguera" temporal del VLM o de una alucinación semántica del LLM.

### 5_iteracion6.ipynb
Clausura metodológica del proyecto mediante ingeniería inversa sobre la memoria visual.
* **Diagnóstico Tabular Supervisado:** Aplana las memorias visuales generadas (JSON) transformándolas en un espacio matricial clásico (variables categóricas, confianza y visibilidad). Se entrenan modelos de Regresión Logística y Random Forest para determinar empíricamente el techo predictivo del sistema sin la intervención de LLMs.
* **Poda Dirigida:** Se aplica un estudio de ablación algorítmica para suprimir las variables que introducen ruido en la función de coste.
* **Cierre Binario (1 vs 3):** Se descartan los lanzamientos por el centro (Zona 2) debido a su alta entropía biomecánica. Se fuerza al sistema a discriminar exclusivamente entre un tiro cruzado y un tiro abierto, logrando aislar la capacidad pura del sistema para leer la lateralidad de la cadena cinética.

## Tecnologías y Dependencias

El pipeline está diseñado para operar en un entorno de ejecución local, garantizando la privacidad de los datos y la ausencia de costes por llamadas a APIs comerciales.

**Stack Tecnológico Principal:**
* `Python 3.x`: Lenguaje base del proyecto.
* `pandas` y `numpy`: Ingeniería de características, transformaciones tabulares y manipulación de estructuras de datos.
* `scikit-learn`: Implementación de tuberías de preprocesamiento (imputación, escalado, codificación One-Hot), validación cruzada estratificada, modelos de diagnóstico (Regresión Logística, Random Forest) y extracción de métricas.
* `opencv-python`: Procesamiento computacional de los archivos de vídeo (.mp4), recorte vectorial y extracción de fotogramas.
* `matplotlib` y `seaborn`: Renderizado de gráficas estadísticas, mapas de calor para matrices de confusión y análisis de distribuciones espaciales.
* `ultralytics` (YOLOv8): Estimación de pose bidimensional y superposición esquelética.

**Servidor de Inferencia Local:**
* **LM Studio:** Motor empleado para la ejecución de los modelos fundacionales. Se utiliza para cargar los pesos de los modelos (formato cuantizado GGUF, como `Q6_K`) y servirlos a través de un endpoint local compatible con los estándares de solicitud HTTP. 

## Instrucciones de Reproducción

1. Clone el repositorio en su máquina local.
2. Asegúrese de contar con un entorno virtual configurado con las bibliotecas listadas en el apartado de tecnologías.
3. Descargue LM Studio, instale los modelos especificados en los cuadernos (con soporte para visión en las fases correspondientes) e inicie el servidor local en el puerto por defecto (1234).
4. En el archivo `1_preprocesamiento.ipynb`, modifique la variable `base_path` para que apunte a su directorio local donde residen los vídeos brutos y el archivo CSV de etiquetas maestras.
5. Ejecute los cuadernos en estricto orden secuencial. Los artefactos intermedios (vídeos procesados, memorias visuales CSV y matrices de confusión) se generarán y guardarán automáticamente en los directorios de salida correspondientes.
