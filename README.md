# Prueba Técnica: Detección de Fraccionamiento Transaccional

Este documento describe el proceso seguido para **explorar** los datos, **identificar** variables relevantes y **definir** un modelo/heurístico para detectar la práctica del **Fraccionamiento Transaccional**, entendida como la división de una transacción grande en varias más pequeñas dentro de una misma ventana de 24 horas.

---

## Tabla de Contenido
1. [Introducción](#introducción)  
2. [Objetivo](#objetivo)  
3. [Alcance y Definiciones](#alcance-y-definiciones)  
4. [Exploración y Evaluación de Datos (EDA)](#exploración-y-evaluación-de-datos-eda)  
   - [Calidad de Datos](#calidad-de-datos)  
   - [Estadísticas Descriptivas](#estadísticas-descriptivas)  
   - [Hipótesis Iniciales](#hipótesis-iniciales)  
5. [Definición del Modelo Analítico](#definición-del-modelo-analítico)  
   - [Flujo de Datos](#flujo-de-datos)  
   - [Lógica de Fraccionamiento](#lógica-de-fraccionamiento)  
   - [Frecuencia de Actualización](#frecuencia-de-actualización)  
   - [Arquitectura Ideal (Opcional)](#arquitectura-ideal-opcional)  
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

3. **Consistencia en Fechas**
   - Rango de fechas plausible (ej.: desde 2021-01-01 hasta 2021-11-30, sin fechas futuras ni inválidas).

### Estadísticas Descriptivas
- **Montos (`transaction_amount`)**  
  - Mínimo: 0.10  
  - Máximo: ~5000.00  
  - Media: ~250.50  
  - Mediana: ~120.00  
  - Desviación Estándar: ~300.00

- **Distribución por Tipo (`transaction_type`)**  
  - ~65% DÉBITO  
  - ~35% CRÉDITO

#### Visualizaciones (si aplica)
- **Histograma** de montos mostró mayor densidad en valores < 500.  
- **Boxplot** detectó algunos outliers en ~4000-5000. Se consideró analizar más a detalle.

### Hipótesis Iniciales
1. **Fraccionamiento por Conteo**: Si un usuario hace >= 3 transacciones pequeñas en 24h, podría ser fraccionamiento.  
2. **Fraccionamiento por Suma**: Varias transacciones que juntas superen un valor típico de “transacción grande” (ej.: 1000).  
3. **Mismo Usuario / Cuenta**: Se asume que las transacciones deben compartir `user_id` o `account_number` para considerarse parte del mismo fraccionamiento.

---

## Definición del Modelo Analítico

### Flujo de Datos
1. **Ingesta**  
   - Se reciben los datos (CSV / Core Financiero).
2. **Preprocesamiento**  
   - Limpieza, formateo de fechas, cálculo de campos derivados.
3. **Generación de Atributos (Features)**  
   - Conteo de transacciones en 24h, suma de montos, etc.
4. **Aplicación de la Lógica**  
   - Se etiquetan las transacciones como “fraccionadas” o “no fraccionadas” según la ventana de 24h.
5. **Salida**  
   - Creación de un flag (`fraction_flag`) para cada transacción.

### Lógica de Fraccionamiento
- **Ventana**: `[t - 24h, t]` para cada transacción.  
- **Criterio**:  
  - Si en esa ventana hay >= 2 transacciones (puede ajustarse a 3, 4, etc.), marcar como `fraccionada`.

```python
def check_fraccionamiento_24h(user_df, min_count=2):
    times = user_df['transaction_date'].values
    flags = []
    
    for i in range(len(times)):
        current_time = times[i]
        window_mask = (
            (times >= current_time - np.timedelta64(24, 'h')) &
            (times <= current_time)
        )
        window_sub = user_df[window_mask]
        count_tx = len(window_sub)
        
        flags.append(count_tx >= min_count)
    
    return flags
