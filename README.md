# Prueba Técnica: Detección de Fraccionamiento Transaccional

Este documento describe el proceso seguido para **explorar** los datos, **identificar** variables relevantes y **definir** un modelo/heurístico para detectar la práctica del **Fraccionamiento Transaccional**, entendida como la división de una transacción grande en varias más pequeñas dentro de una misma ventana de 24 horas.

---

## Tabla de Contenido
1. [Introducción](#introducción)  
2. [Objetivo](#objetivo)  
3. [Alcance y Definiciones](#alcance-y-definiciones)  
4. [Exploración y Evaluación de Datos (EDA)](#exploración-y-evaluación-de-datos-eda)  
   - [Calidad de Datos](#calidad-de-datos)  
   - [Estadísticos Descriptivos](#estadísticos-descriptivos)  
   - [Hipótesis Iniciales](#hipótesis-iniciales)  
5. [Definición del Modelo Analítico](#definición-del-modelo-analítico)  
   - [Flujo de Datos](#flujo-de-datos)  
   - [Lógica de Fraccionamiento](#lógica-de-fraccionamiento)  
   - [Frecuencia de Actualización](#frecuencia-de-actualización)  
   - [Arquitectura Ideal](#arquitectura-ideal)
6. [Conclusiones y Recomendaciones](#conclusiones-y-recomendaciones)

---

## Introducción
En el marco de la prueba técnica, se facilita un conjunto de datos (`dataset`) que contiene información de transacciones financieras. El desafío consiste en **detectar** la mala práctica de **Fraccionamiento Transaccional**, donde un usuario (o cuenta) realiza múltiples transacciones de menor valor que, agrupadas en una ventana de 24 horas, equivalen o superan una supuesta “transacción original”.

---

## Objetivo
1. **Identificar** patrones de fraccionamiento transaccional en el dataset provisto.
2. **Describir** el proceso analítico, desde la exploración de datos (EDA) hasta la propuesta de modelo o reglas.  
3. **Proveer** una recomendación sobre la frecuencia de actualización y, opcionalmente, diseñar una arquitectura para desplegar la solución.

---

## Alcance y Definiciones
- **Fraccionamiento Transaccional**: Práctica de dividir un monto grande en varios montos pequeños dentro de un lapso de 24 horas, con la misma cuenta o el mismo usuario.
- **Ventana de Tiempo (24 horas)**: Período en el cual se evalúa la suma y el conteo de transacciones para identificar la fracción de un monto mayor.

### Columnas relevantes
- **`_id`**: Identificador único del registro.
- **`merchant_id`**: Código único del comercio o aliado.
- **`subsidiary`**: Código único de la sede o sucursal.
- **`transaction_date`**: Fecha de transacción en el core financiero.
- **`account_number`**: Número único de cuenta.  
- **`user_id`**: Código único del usuario dueño de la cuenta desde donde se registran las transacciones.
- **`transaction_amount`**: Monto de la transacción (en moneda ficticia).
- **`transaction_type`**: Naturaleza de la transacción (crédito o débito).

---

## Exploración y Evaluación de Datos (EDA)
Durante esta fase, se realizaron **descriptivos** y **validaciones** de calidad para comprender el comportamiento de las transacciones.

### Muestreo de los Datos
Se trabajó con una muestra correspondiente al 10% del tamaño de los datasets originales.

### Calidad de Datos
1. **Valores Nulos / Faltantes**  
   - No se identificó valores nulos a nivel de registro.

2. **Duplicados**  
   - Al analizar la **relación** entre `user_id` y `account_number`, no se cumple la condición de 1:1.
        - Es decir, un mismo `account_number` puede estar asociado a varios `user_id`, reflejando cuentas compartidas o varios autorizados sobre una sola cuenta. 
        - Esta situación no se considera un “duplicado” tradicional, pero sí indica que distintas personas (o identificadores) podrían usar la misma cuenta, lo cual puede influir en el análisis de fraccionamiento (p. ej., hay que decidir si se agrupa por usuario o por cuenta, o por ambos).

3. **Consistencia en fechas**
   - Rango de fechas plausible (ej.: desde 2021-01-01 hasta 2021-11-30, sin fechas futuras ni inválidas).

### Estadísticos Descriptivas
- **Montos (`transaction_amount`)**  
  - Mínimo: 5.94  
  - Máximo: ~3,210  
  - Media: ~191
  - Mediana: 107  
  - Desviación Estándar: ~241

- **Distribución por tipo (`transaction_type`)**  
  - 80% Débito  
  - 20% Crédito

#### Visualizaciones
- **Histograma**
   - La mayoría de las transacciones tienen montos bajos (concentradas cerca de 0).
   - Pocas transacciones tienen montos altos, lo que genera una larga cola hacia la derecha.

![Histograma](./reports/fig1.png "Histograma")

- **Boxplot**
   - La gran cantidad de valores atípicos sugiere que el conjunto de datos tiene muchas transacciones poco comunes con montos elevados.
   - La mediana está cerca del límite inferior de la caja, lo que refuerza que la mayoría de los datos están concentrados en valores bajos.

![Boxplot](./reports/fig2.png "Boxplot")

### Hipótesis Iniciales
1. **Fraccionamiento por conteo**: Si un usuario hace más de 2 transacciones pequeñas en 24h, podría ser considerado como fraccionamiento.
2. **Mismo Usuario / Cuenta**: Se asume que las transacciones deben compartir `user_id` o `account_number` para considerarse parte del mismo fraccionamiento.

---

## Definición del Modelo Analítico

### Flujo de Datos
1. **Ingesta**  
   - Se reciben los datos en cualquier tipo de formato (p. ej. parquet, csv, etc).
2. **Preprocesamiento**  
   - Limpieza, formateo de fechas, eliminación de registros duplicados, entre otros.
3. **Generación de Atributos (Features)**  
   - Conteo de transacciones en 24h, suma de montos, etc.
4. **Aplicación de la Lógica (Regla de negocios)**  
   - Se etiquetan las transacciones como “fraccionadas” o “no fraccionadas” según la ventana de 24h.
5. **Salida**  
   - Creación de un flag (`fraction_flag`) para cada transacción.

### Criterio de Selección del Modelo Analítico

El **modelo analítico** propuesto consiste en la **evaluación de transacciones** dentro de una **ventana rodante de 24 horas** y la aplicación de una **regla heurística** (conteo de transacciones en ese lapso). A continuación, se describe el **porqué** de esta selección:

1. **Simplicidad e Interpretabilidad**  
   - Al basarse en una regla de conteo (por ejemplo, “si hay *n* transacciones en 24 horas, marcar fraccionamiento”), es **fácil de entender y explicar** para el equipo de negocio, auditoría o cumplimiento.  
   - No requiere conocimientos avanzados de estadísticas o machine learning; además, **las alertas generadas** son trazables a un criterio claro.

2. **Rapidez de Implementación**  
   - Los **umbrales** (número mínimo de transacciones o sumas de montos) se pueden ajustar rápidamente con base en la experiencia de negocio.  
   - El código puede implementarse en `pandas`, `SQL` o un motor de reglas sin una curva de aprendizaje compleja.

3. **Escalabilidad**  
   - Aunque en un escenario de alto volumen podría requerirse optimización (Spark, Dask, etc.), la **idea principal** (ventanas de 24h, conteo de transacciones) es sencilla de escalar.  
   - También es **fácil de trasladar** a soluciones de streaming o batch según la necesidad.

4. **Modelos más sofisticados**
   - Esta **regla heurística** puede servir como **base** para luego implementar enfoques de Machine Learning (como detección de anomalías) o modelos estadísticos más avanzados.  
   - Permite crear un **punto de partida** e iterar en la medida que la organización requiera mayor precisión o menos falsos positivos.

En síntesis, se eligió un **modelo heurístico** y **reglas de negocio** por su **clara interpretabilidad, bajo costo de implementación** y **alineación con las definiciones de fraccionamiento transaccional** (ventanas de 24 horas, número mínimo de transacciones), lo que facilita la **adopción y validación** en entornos de cumplimiento o auditoría.

### Lógica de Fraccionamiento
- **Ventana**: `[t - 24h, t]` para cada transacción.  
- **Criterio**:  
  - Si en esa ventana hay >= 2 transacciones (puede ajustarse a 3, 4, etc.), marcar como **fraccionada**.

```python
# Función que aplica las reglas de negocio
def check_fraccionamiento_24h(user_df, min_count=2):
    times = user_df['transaction_date'].values
    flags = list()
    for i in range(len(times)):
        current_time = times[i]
        window_mask = (
            (times >= current_time - np.timedelta64(24, 'h')) & 
            (times <= current_time)
        )
        window_sub = user_df[window_mask]
        count_tx = len(window_sub)
        
        if count_tx >= min_count:
            flags.append(True)
        else:
            flags.append(False)
            
    return flags

# 1) Ordenar el DataFrame
data = data.sort_values(by=['user_id', 'transaction_date'])

# 2) Generar el array de valores booleanos sin asignarlos al DataFrame todavía
bool_flags = (
    data
    .groupby('user_id', group_keys=False)
    .apply(lambda df: check_fraccionamiento_24h(df, min_count=2))
    .explode()        # Separa la lista booleana por filas
    .astype(bool)     # Asegurarnos de que sea tipo bool
    .values           # Convertir a array de numpy
)

# 3) Crear directamente la columna de texto 'fraction_flag'
data['fraction_flag'] = np.where(
    bool_flags,
    'FRACCIONADA', 
    'NO_FRACCIONADA'
)
```

### Frecuencia de Actualización
Dado el **caso de estudio** y las consideraciones previas (ventaja de simplicidad vs. inmediatez en la detección), se sugiere un **enfoque híbrido** que equilibre la rapidez de respuesta y la factibilidad técnica:

1. **Ejecución diaria (batch) como base**  
   - A nivel **operativo**, se programa una corrida cada noche para consolidar y analizar todas las transacciones del día, marcando aquellas que cumplan con los criterios de fraccionamiento.  
   - **Ventajas**:
     - Implica **menor complejidad** de desarrollo e infraestructura.
     - Ofrece un **análisis exhaustivo** con la ventana de 24 horas completa y cerrada.

2. **Alertas en tiempo casi real (Streaming) para casos críticos**  
   - Si se requiere actuar sobre **montos muy altos** o ciertos indicadores de riesgo, implementar **reglas de streaming** enfocadas solo en **eventos de alto impacto**.  
   - **Ventajas**:
     - Evita la necesidad de procesar absolutamente todas las transacciones en streaming.
     - Permite **detección inmediata** de situaciones realmente críticas (p. ej., muchas transacciones en minutos consecutivos, sumas muy elevadas, etc.).

#### Justificación

- **Coste y Complejidad**: Un sistema 100% en streaming puede resultar costoso e innecesario si la mayoría de casos no requieren respuesta inmediata.  
- **Rapidez de Implementación**: Al tener un proceso batch como núcleo, el desarrollo inicial es más sencillo y estable.  
- **Análisis Integral**: La corrida nocturna garantiza un **reporte completo** de fraccionamientos para todo el día, que puede revisarse por áreas de cumplimiento o riesgo.  
- **Escalabilidad**: En caso de que aparezcan más casos urgentes o se exija una respuesta aún más rápida, se pueden **ampliar** gradualmente las reglas de streaming.

En conclusión, se puede combinar un **proceso batch diario** con una **capa de streaming** enfocada en eventos de riesgo alto, maximizando la relación **costo–beneficio** y proporcionando, a la vez, la **reacción inmediata** en los escenarios más críticos.
