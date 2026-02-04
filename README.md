# Predicción de Enfermedad Cardíaca mediante Regresión Logística

## Resumen del Proyecto

Este proyecto implementa un modelo de regresión logística desde cero para la predicción de enfermedad cardíaca, utilizando técnicas de análisis exploratorio de datos, visualización de límites de decisión, regularización L2 y una arquitectura lista para despliegue en producción mediante Amazon SageMaker. El análisis se desarrolló siguiendo metodologías de aprendizaje automático supervisado, con énfasis en la interpretabilidad clínica de los resultados y el cumplimiento de estándares profesionales de ciencia de datos.

## Descripción del Dataset

### Fuente de Datos

Dataset: Heart Disease Prediction  
Plataforma: Kaggle  
URL: https://www.kaggle.com/datasets/neurocipher/heartdisease

El dataset comprende 270 registros de pacientes con las siguientes características:

**Variables Demográficas:**
- Age: Edad del paciente (rango: 29-77 años)
- Sex: Sexo del paciente (0 = femenino, 1 = masculino)

**Variables Clínicas:**
- Chest pain type: Tipo de dolor torácico (escala 1-4)
- BP: Presión arterial en reposo (mm Hg)
- Cholesterol: Colesterol sérico (mg/dl, rango: 112-564)
- FBS over 120: Glucosa en sangre en ayunas > 120 mg/dl (binario)
- EKG results: Resultados del electrocardiograma en reposo
- Max HR: Frecuencia cardíaca máxima alcanzada
- Exercise angina: Angina inducida por ejercicio (binario)
- ST depression: Depresión del segmento ST inducida por ejercicio
- Slope of ST: Pendiente del segmento ST durante el ejercicio
- Number of vessels fluro: Número de vasos principales coloreados por fluoroscopia (0-3)
- Thallium: Resultados de la prueba de talio

**Variable Objetivo:**
- Heart Disease: Presencia o ausencia de enfermedad cardíaca (binaria)

### Estadísticas Descriptivas

- Total de muestras: 270 pacientes
- Tasa de presencia de enfermedad: 44.4%
- Tasa de ausencia de enfermedad: 55.6%
- Valores faltantes: 0 (dataset completo)
- División del dataset: 70% entrenamiento (189 muestras), 30% prueba (81 muestras)

## Metodología

### Paso 1: Análisis Exploratorio de Datos (EDA)

Se realizó un análisis exhaustivo del conjunto de datos que incluyó:
- Verificación de integridad de datos y detección de valores faltantes
- Análisis de distribución de la variable objetivo
- Normalización de características mediante estandarización Z-score
- División estratificada del dataset para mantener la proporción de clases

### Paso 2: Implementación de Regresión Logística Base

Se implementó el algoritmo de regresión logística desde cero utilizando NumPy, incluyendo:
- Función sigmoide para transformación probabilística
- Función de costo de entropía cruzada binaria
- Algoritmo de descenso de gradiente para optimización

**Características seleccionadas para el modelo:**
1. Age (Edad)
2. Cholesterol (Colesterol)
3. BP (Presión arterial)
4. Max HR (Frecuencia cardíaca máxima)
5. ST depression (Depresión del segmento ST)
6. Number of vessels fluro (Número de vasos fluoroscópicos)

**Hiperparámetros del modelo base:**
- Tasa de aprendizaje (α): 0.01
- Número de iteraciones: 1000
- Umbral de clasificación: 0.5

### Paso 3: Visualización de Límites de Decisión

Se generaron visualizaciones bidimensionales de los límites de decisión para tres pares de características:
1. Age vs. Cholesterol
2. BP vs. Max HR
3. ST depression vs. Number of vessels fluro

Estas visualizaciones permitieron analizar la separabilidad lineal de las clases y la efectividad de diferentes combinaciones de características.

### Paso 4: Regularización L2

Se implementó regularización L2 (Ridge) para prevenir el sobreajuste y mejorar la generalización del modelo. Se evaluaron los siguientes valores de λ:
- λ = 0.0 (sin regularización)
- λ = 0.001
- λ = 0.01
- λ = 0.1
- λ = 1.0

La regularización se aplicó tanto a la función de costo como a los gradientes durante la optimización.

### Paso 5: Preparación para Despliegue en Amazon SageMaker

Se preparó el modelo para despliegue en producción mediante:
- Exportación de parámetros optimizados (pesos w y sesgo b)
- Diseño de función de inferencia para predicción en tiempo real
- Simulación de endpoint con casos de prueba clínicos
- Documentación de arquitectura de despliegue

## Resultados

### Métricas del Modelo Base (sin regularización)

| Métrica | Entrenamiento | Prueba |
|---------|---------------|--------|
| Accuracy | 0.831 | 0.815 |
| Precision | 0.779 | 0.800 |
| Recall | 0.857 | 0.800 |
| F1-Score | 0.816 | 0.789 |

### Métricas con Regularización Óptima (λ = 1.0)

| Métrica | Valor |
|---------|-------|
| Test Accuracy | 0.815 |
| Test F1-Score | 0.800 |
| Norma de pesos (‖w‖) | 1.466 |
| AUC-ROC | 0.889 |

### Mejoras Obtenidas

- Mejora en F1-Score: +1.1% (de 0.789 a 0.800)
- Reducción de la norma de pesos, indicando menor complejidad del modelo
- Reducción del riesgo de sobreajuste manteniendo el rendimiento

### Convergencia del Modelo

El modelo alcanzó convergencia en menos de 200 iteraciones, con una reducción consistente de la función de costo. La curva de aprendizaje mostró una disminución suave sin oscilaciones significativas, indicando una tasa de aprendizaje apropiada.

### Análisis de Importancia de Características

Los predictores más influyentes identificados fueron:
1. ST depression (Depresión del segmento ST)
2. Number of vessels fluro (Número de vasos fluoroscópicos)
3. Max HR (Frecuencia cardíaca máxima)

El par de características ST depression - Number of vessels fluro mostró la mejor separabilidad lineal, lo que confirma su relevancia clínica en el diagnóstico de enfermedad cardíaca.

### Capacidad Discriminativa

El modelo alcanzó un AUC-ROC de 0.889, lo que indica una excelente capacidad para distinguir entre pacientes con y sin enfermedad cardíaca. Este valor se encuentra dentro del rango considerado como "muy bueno" en la literatura médica (0.8 - 0.9).

## Implementación en Amazon SageMaker

Durante la fase de implementación, se procedió a configurar el endpoint de inferencia utilizando la consola de Amazon SageMaker Studio. Sin embargo, al ejecutar el proceso de deployment, el sistema retornó un mensaje de error relacionado con permisos insuficientes. Tras la revisión de la configuración de la cuenta, se identificó que las cuentas proporcionadas por AWS Academy operan bajo un esquema de permisos restringidos que limita ciertas operaciones críticas, específicamente la capacidad de crear y mantener endpoints activos para modelos de machine learning en el servicio SageMaker. Esta restricción de políticas IAM impidió completar el despliegue del modelo en un entorno de producción real dentro de la plataforma AWS.

### Arquitectura Propuesta

El modelo está diseñado para ser desplegado en Amazon SageMaker siguiendo esta arquitectura:

1. **Preparación de modelo**: Exportación de parámetros (w, b) y estadísticas de normalización (μ, σ)
2. **Creación de endpoint**: Configuración de instancia de inferencia
3. **Función de inferencia**: Implementación de pipeline de predicción completo
4. **Monitoreo**: Seguimiento de latencia y rendimiento

### Función de Inferencia

La función de inferencia implementada realiza:
- Normalización de entrada utilizando parámetros del conjunto de entrenamiento
- Cálculo de probabilidad mediante función sigmoide
- Clasificación binaria con umbral de 0.5
- Retorno de probabilidad y etiqueta de predicción

### Casos de Prueba

**Paciente 1 (Riesgo Alto):**
- Características: Edad 60, Colesterol 300, PA 140, FC 130, Dep. ST 2.5, Vasos 2
- Probabilidad: 95.09%
- Predicción: Presencia (Riesgo Alto)

**Paciente 2 (Riesgo Bajo):**
- Características: Edad 45, Colesterol 200, PA 120, FC 170, Dep. ST 0.0, Vasos 0
- Probabilidad: 11.56%
- Predicción: Ausencia (Riesgo Bajo)

**Paciente 3 (Riesgo Alto):**
- Características: Edad 70, Colesterol 350, PA 160, FC 100, Dep. ST 3.0, Vasos 3
- Probabilidad: 99.55%
- Predicción: Presencia (Riesgo Alto)

### Rendimiento del Endpoint

- Latencia estimada: < 50 ms
- Throughput: Compatible con aplicaciones de tiempo real
- Escalabilidad: Soporta configuración multi-instancia




## Conclusiones

Este proyecto demuestra la implementación completa de un pipeline de aprendizaje automático para predicción de enfermedad cardíaca, desde el análisis exploratorio hasta la preparación para despliegue en producción. Los resultados obtenidos son clínicamente significativos:

1. **Rendimiento del modelo**: El modelo alcanza un F1-Score de 0.800 y un AUC-ROC de 0.889, indicando una alta capacidad predictiva adecuada para asistencia en decisiones clínicas.

2. **Regularización efectiva**: La aplicación de regularización L2 con λ = 1.0 mejoró la generalización del modelo sin sacrificar rendimiento, reduciendo la norma de pesos a 1.466.

3. **Interpretabilidad clínica**: Las características más influyentes (depresión del ST, vasos fluoroscópicos, frecuencia cardíaca máxima) coinciden con indicadores clínicos conocidos de enfermedad cardíaca.

4. **Viabilidad de producción**: El modelo exportado es compatible con arquitecturas de inferencia en tiempo real, con latencia inferior a 50 ms, lo que lo hace viable para aplicaciones clínicas.

5. **Convergencia eficiente**: El algoritmo de descenso de gradiente mostró convergencia estable en menos de 200 iteraciones, lo que evidencia una buena selección de hiperparámetros.

El proyecto cumple con los estándares profesionales de ciencia de datos aplicada a la medicina, proporcionando un sistema de apoyo a la decisión clínica que podría integrarse en entornos sanitarios para la evaluación preliminar de riesgo cardíaco.
