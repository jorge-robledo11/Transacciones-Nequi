# • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ Pyfunctions ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ •
# Librerías y/o depedencias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import empiricaldist
from IPython.display import display, Latex
from scipy import stats
from matplotlib import gridspec
from matplotlib.dates import DateFormatter, HourLocator
from sklearn.preprocessing import PowerTransformer
sns.set_theme(context='notebook', style=plt.style.use('dark_background')) # type: ignore


# Capturar variables
# Función para capturar los tipos de variables
def capture_variables(data: pd.DataFrame) -> tuple[list]:
    
    """
    Function to capture the types of Dataframe variables

    Args:
        dataframe: DataFrame
    
    Return:
        variables: A tuple of lists
    
    The order to unpack variables:
    1. continous
    2. categoricals
    3. discretes
    4. temporaries
    """

    numericals = list(data.select_dtypes(include = [np.int32, np.int64, np.float32, np.float64]).columns) # type: ignore
    categoricals = list(data.select_dtypes(include = ['category', 'object', 'bool']).columns)
    temporaries = list(data.select_dtypes(include = ['datetime', 'timedelta']).columns)
    discretes = [col for col in data[numericals] if len(data[numericals][col].unique()) <= 10]
    continuous = [col for col in data[numericals] if col not in discretes]

    # Variables
    print('\t\tTipos de variables')
    print(f'Hay {len(continuous)} variables continuas')
    print(f'Hay {len(discretes)} variables discretas')
    print(f'Hay {len(temporaries)} variables temporales')
    print(f'Hay {len(categoricals)} variables categóricas')
    
    variables = tuple((continuous, categoricals, discretes, temporaries))

    # Retornamos una tupla de listas
    return variables


# Función para obtener la matriz de correlaciones entre los predictores
def correlation_matrix(data:pd.DataFrame, continuous: list) -> None:
    
    """
    Function to plot correlation_matrix

    Args:
        data: DataFrame
        continuous: list
    
    Return:
        Dataviz
    """
    
    correlations = data[continuous].corr(method='pearson', numeric_only=True)
    plt.figure(figsize=(14, 7))
    sns.heatmap(correlations, vmax=1, annot=True, cmap='gist_yarg', linewidths=1, square=True)
    plt.title('Matriz de Correlaciones\n', fontsize=14)
    plt.xticks(fontsize=10, rotation=25)
    plt.yticks(fontsize=10, rotation=25)
    plt.tight_layout()
    

# Covarianza entre los predictores
# Función para obtener una matriz de covarianza con los predictores
def covariance_matrix(data:pd.DataFrame):
    
    """
    Function to get mapped covariance matrix

    Args:
        data: DataFrame
    
    Return:
        DataFrame
    """
    
    cov_matrix = data.cov()
    
    # Crear una matriz de ceros con el mismo tamaño que la matriz de covarianza
    zeros_matrix = np.zeros(cov_matrix.shape)
    
    # Crear una matriz diagonal de ceros reemplazando los valores de la diagonal de la matriz con ceros
    diagonal_zeros_matrix = np.diag(zeros_matrix)
    
    # Reemplazar la diagonal de la matriz de covarianza con la matriz diagonal de ceros
    np.fill_diagonal(cov_matrix.to_numpy(), diagonal_zeros_matrix)
    
    # Mapear los valores con etiquetas para saber cómo covarian los predictores
    cov_matrix = cov_matrix.map(lambda x: 'Positivo' if x > 0 else 'Negativo' if x < 0 else '') # type: ignore
    
    return cov_matrix


# Función para graficar la covarianza entre los predictores
def plotting_covariance(X:pd.DataFrame, continuous: list, n_iter: int) -> None:
    
    """
    Function to plot covariance matrix choosing some random predictors

    Args:
        X: DataFrame
        continuous: list
        n_iter: int

    Return:
        DataViz
    """

    # Semilla para efectos de reproducibilidad
    np.random.seed(n_iter)

    for _ in range(n_iter):
        # Creamos una figura con tres subfiguras
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        plt.suptitle('Covariance Plots\n', fontsize=15)

        # Seleccionamos dos variables aleatorias del Dataframe
        var1 = np.random.choice(X[continuous].columns)
        var2 = np.random.choice(X[continuous].columns)
        while var1 == var2:
            var2 = np.random.choice(X[continuous].columns)

        # Graficamos la covarianza en la primera subfigura
        sns.scatterplot(x=var1, y=var2, data=X[continuous], ax=ax1, color='red', alpha=0.6)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.xticks()
        plt.yticks()
        ax1.grid(color='white', linestyle='-', linewidth=0.25)

        # Seleccionamos dos nuevas variables aleatorias del Dataframe
        var1 = np.random.choice(X[continuous].columns)
        var2 = np.random.choice(X[continuous].columns)
        while var1 == var2:
            var2 = np.random.choice(X[continuous].columns)

        # Graficamos la covarianza en la segunda subfigura
        sns.scatterplot(x=var1, y=var2, data=X[continuous], ax=ax2, color='green', alpha=0.6)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.xticks()
        plt.yticks()
        ax2.grid(color='white', linestyle='-', linewidth=0.25)

        # Seleccionamos otras dos variables aleatorias del Dataframe
        var1 = np.random.choice(X[continuous].columns)
        var2 = np.random.choice(X[continuous].columns)
        while var1 == var2:
            var2 = np.random.choice(X[continuous].columns)

        # Graficamos la covarianza en la tercera subfigura
        sns.scatterplot(x=var1, y=var2, data=X[continuous], ax=ax3, color='blue', alpha=0.6)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.xticks()
        plt.yticks()
        ax3.grid(color='white', linestyle='-', linewidth=0.25)

        # Mostramos la figura
        fig.tight_layout()
        plt.show()


# Función para obtener la estratificación de clases/target
def class_distribution(data: pd.DataFrame, target: str) -> None:
    """
    Function to get balance by classes

    Args:
        data: DataFrame
        target: str
    
    Return:
        Dataviz
    """

    # Distribución de clases
    distribucion = data[target].value_counts(normalize=True)

    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 4))

    # Ajustar el margen izquierdo de los ejes para separar las barras del eje Y
    ax.margins(y=0.2)

    # Ajustar la posición de las etiquetas de las barras
    ax.invert_yaxis()

    # Crear gráfico de barras horizontales con la paleta de colores personalizada
    ax.barh(distribucion.index, distribucion.values, align='center', color='lightblue', # type: ignore
            edgecolor='white', height=0.5, linewidth=0.5, alpha=0.7)

    # Definir título y etiquetas de los ejes
    ax.set_title('Distribución de clases\n', fontsize=14)
    ax.set_xlabel('Porcentajes', fontsize=12)
    ax.set_ylabel(f'{target}'.capitalize(), fontsize=12)

    # Configurar los ticks del eje Y para que solo muestren 0 y 1
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', '1'])

    # Mostrar el gráfico
    plt.grid(color='white', linestyle='-', linewidth=0.25)
    plt.tight_layout()
    plt.show()
    

def plot_variable_behaviors_over_time(df: pd.DataFrame, time_col: str, target_col: str):
    """
    Plots the behavior of variables over time for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        time_col (str): The name of the column representing time.
        target_col (str): The target variable to exclude from plotting.
    
    Returns:
        None: Displays the plots.
    """
    # Ordenar el DataFrame por la columna temporal si no está ordenado
    df = df.sort_values(time_col)
    
    # Seleccionar las variables a graficar (excluyendo la columna temporal y el target)
    variables = [col for col in df.columns if col != time_col and col != target_col]
    num_variables = len(variables)

    # Configurar la figura
    plt.figure(figsize=(15, 5 * num_variables))  # Ajustar altura de cada subgráfico

    for i, var in enumerate(variables):
        plt.subplot(num_variables, 1, i + 1)
        plt.plot(df[time_col], df[var], label=f'Comportamiento de {var}', color='#7dcea0')
        plt.title(f'Comportamiento de {var} a lo largo del tiempo\n', fontsize=14)
        plt.xlabel(f'Tiempo ({time_col})', fontsize=12)
        plt.ylabel(var, fontsize=12)
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        
        # Ajustar tamaño de fuente de la leyenda
        plt.legend(fontsize=10)
        
        # Personalizar eje X para más fechas en los ticks
        ax = plt.gca()
        ax.xaxis.set_major_locator(HourLocator(interval=6))  # Mostrar ticks cada 6 horas
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))  # Formato granular
        
        # Ajustar tamaño de fuente de xticks
        plt.xticks(fontsize=10, rotation=45)  # Rotar etiquetas para mejor visualización

    # Ajustar diseño para evitar solapamiento
    plt.tight_layout()
    plt.show()


# Diagnóstico de variables
# Función para observar el comportamiento de variables continuas
def diagnostic_plots(data:pd.DataFrame, variables: list[str]) -> None:

    """
    Function to get diagnostic graphics into 
    numerical (continous and discretes) predictors

    Args:
        data: DataFrame
        variables: list
    
    Return:
        Dataviz
    """
        
    for var in data[variables]:
        fig, (ax, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle('Diagnostic Plots', fontsize=16)
        plt.rcParams.update({'figure.max_open_warning': 0}) # Evitar un warning

        # Histogram Plot
        plt.subplot(1, 4, 1)
        plt.title('Histogram Plot')
        sns.histplot(data[var], bins=25, color='midnightblue', edgecolor='white', lw=0.5) # type: ignore
        plt.axvline(data[var].mean(), color='#E51A4C', ls='dashed', lw=1.5, label='Mean')
        plt.axvline(data[var].median(), color='gold', ls='dashed', lw=1.5, label='Median')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=25)
        plt.xlabel(str(var))
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        plt.legend(fontsize=10)
        
        # CDF Plot
        plt.subplot(1, 4, 2)
        plt.title('CDF Plot')
        xs = np.linspace(data[var].min(), data[var].max())
        ys = stats.norm(data[var].mean(), data[var].std()).cdf(xs) # Distribución normal a partir de unos datos
        plt.plot(xs, ys, color='cornflowerblue', ls='dashed')
        empiricaldist.Cdf.from_seq(data[var], normalize=True).plot(color='chartreuse')
        plt.xlabel(str(var))
        plt.xticks(rotation=25)
        plt.ylabel('Probabilidad')
        plt.legend(['Distribución normal', var], fontsize=8, loc='upper left')
        plt.grid(color='white', linestyle='-', linewidth=0.25)

        # PDF Plot
        plt.subplot(1, 4, 3)
        plt.title('PDF Plot')
        kurtosis = stats.kurtosis(data[var], nan_policy='omit') # Kurtosis
        skew = stats.skew(data[var], nan_policy='omit') # Sesgo
        xs = np.linspace(data[var].min(), data[var].max())
        ys = stats.norm(data[var].mean(), data[var].std()).pdf(xs) # type: ignore
        plt.plot(xs, ys, color='cornflowerblue', ls='dashed')
        sns.kdeplot(data=data, x=data[var], fill=True, lw=0.75, color='crimson', alpha=0.5, edgecolor='white')
        plt.text(s=f'Skew: {skew:0.2f}\nKurtosis: {kurtosis:0.2f}',
                 x=0.25, y=0.65, transform=ax3.transAxes, fontsize=11,
                 verticalalignment='center', horizontalalignment='center')
        plt.ylabel('Densidad')
        plt.xticks(rotation=25)
        plt.xlabel(str(var))
        plt.xlim()
        plt.legend(['Distribución normal', var], fontsize=8, loc='upper right')
        plt.grid(color='white', linestyle='-', linewidth=0.25)

        # Boxplot & Stripplot
        plt.subplot(1, 4, 4)
        plt.title('Boxplot')
        sns.boxplot(data=data[var], width=0.4, color='silver',
                    boxprops=dict(lw=1, edgecolor='white'),
                    whiskerprops=dict(color='white', lw=1),
                    capprops=dict(color='white', lw=1),
                    medianprops=dict(),
                    flierprops=dict(color='red', lw=1, marker='o', markerfacecolor='red'))
        plt.axhline(data[var].quantile(0.75), color='magenta', ls='dotted', lw=1.5, label='IQR 75%')
        plt.axhline(data[var].median(), color='gold', ls='dashed', lw=1.5, label='Median')
        plt.axhline(data[var].quantile(0.25), color='cyan', ls='dotted', lw=1.5, label='IQR 25%')
        plt.xlabel(str(var))
        plt.tick_params(labelbottom=False)
        plt.ylabel('Unidades')
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        
        fig.tight_layout()
        
        
# Test de Normalidad de D’Agostino y Pearson
# Función para observar el comportamiento de las variables continuas en una prueba de normalidad
def normality_test(data, variables):

    display(Latex('Si el $pvalue$ < 0.05; se rechaza la $H_0$ sugiere que los datos no se ajustan de manera significativa a una distribución normal'))
    
    # Configurar figura
    fig = plt.figure(figsize=(20, 11))
    plt.suptitle('Prueba de Normalidad', fontsize=16)
    gs = gridspec.GridSpec(nrows=len(variables) // 3+1, ncols=3, figure=fig)
    
    for i, var in enumerate(variables):

        ax = fig.add_subplot(gs[i//3, i % 3])

        # Gráfico Q-Q
        stats.probplot(data[var], dist='norm', plot=ax)
        ax.set_xlabel(var)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels())
        ax.grid(color='white', linestyle='-', linewidth=0.25)

        # P-value
        p_value = stats.normaltest(data[var])[1]
        ax.text(0.8, 0.9, f"p-value={p_value:0.3f}", transform=ax.transAxes, fontsize=13) 

    plt.tight_layout(pad=3)
    plt.show()


# Definir la transformación de Yeo-Johnson
def yeo_johnson_transform(x):
    transformer = PowerTransformer(method='yeo-johnson')
    return transformer.fit_transform(x.values.reshape(-1, 1)).flatten()


def gaussian_transformation(data: pd.DataFrame, variables: list) -> dict:
    """
    Function to get Gaussian transformations of the variables

    Args:
        data: DataFrame
        variables: list
    
    Return:
        results: dict
    """
    
    # Definir las transformaciones gaussianas a utilizar
    transformaciones_gaussianas = {
        'Log': lambda x: np.log(x + 1e-6) if (x > 0).all() else np.nan,  # Asegurarse de que x sea positivo
        'Sqrt': lambda x: np.sqrt(x + 1e-6) if (x >= 0).all() else np.nan,  # Asegurarse de que x sea no negativo
        'Reciprocal': lambda x: 1 / (x + 1e-6) if (x > 0).all() else np.nan,  # Evitar división por cero o negativos
        'Exp': lambda x: x**2,
        'Yeo-Johnson': yeo_johnson_transform
    }
    
    # Crear un diccionario para almacenar los resultados de las pruebas de normalidad
    results = dict()

    # Iterar a través de las variables y las transformaciones
    for var in data[variables].columns:
        mejores_p_value = -1  # Iniciar con un valor imposible de p-value
        mejor_transformacion = None
        
        for nombre_transformacion, transformacion in transformaciones_gaussianas.items():
            try:
                # Aplicar la transformación a la columna
                variable_transformada = transformacion(data[var])
                
                # Asegurarse de que la transformación no genere NaN
                if not np.isnan(variable_transformada).any():
                    # Calcular el p-value de la prueba de normalidad
                    p_value = stats.normaltest(variable_transformada)[1]
                    
                    # Actualizar el mejor p-value y transformación si es necesario
                    if p_value > mejores_p_value:
                        mejores_p_value = p_value
                        mejor_transformacion = nombre_transformacion
            except Exception as e:
                # Capturar errores en la transformación y continuar
                print(f"Error al transformar {var} con {nombre_transformacion}: {e}")
                continue
        
        # Almacenar el resultado en el diccionario
        results[var] = mejor_transformacion if mejor_transformacion else "No Transformation Found"
        
    return results



# Graficar la comparativa entre las variables originales y su respectiva transformación
def graficar_transformaciones(data: pd.DataFrame, continuous: list, transformacion: dict) -> None:
    
    """
    Function to plot compare Gaussian transformations of the variables and their original state

    Args:
        data: DataFrame
        variables: list
    
    Return:
        Dataviz
    """
    
    # Definir las transformaciones gaussianas a utilizar
    transformaciones_gaussianas = {
        'Log': np.log,
        'Sqrt': np.sqrt, 
        'Reciprocal': lambda x: 1/x, 
        'Exp': lambda x: x**2, 
        'Yeo-Johnson': yeo_johnson_transform
        }
    
    data = data.copy()
    data = data[continuous]
    
    for variable, transformacion_name in transformacion.items():
        # Obtener datos originales
        data_original = data[variable]
        
        # Obtener la transformación correspondiente
        transformacion_func = transformaciones_gaussianas.get(transformacion_name)
        
        # Aplicar transformación
        data_transformada = transformacion_func(data_original)

        # Crear figura con 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Graficar histograma datos originales 
        hist_kws = {'color': 'royalblue', 'lw': 0.5}
        sns.histplot(data_original, ax=ax1, kde=True, bins=50, **hist_kws)
        ax1.set_title('Original')
        ax1.grid(color='white', linestyle='-', linewidth=0.25)

        # Graficar histograma datos transformados
        sns.histplot(data_transformada, ax=ax2, kde=True, bins=50, **hist_kws)
        ax2.set_title(f'{transformacion_name}')
        ax2.grid(color='white', linestyle='-', linewidth=0.25)
        
        # Cambiar color del KDE en ambos gráficos
        for ax in [ax1, ax2]:
            for line in ax.lines:
                line.set_color('crimson')

        # Mostrar figura
        plt.tight_layout()
        plt.show()