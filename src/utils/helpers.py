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
    return variables # type: ignore


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
    

def plot_variable_behaviors_over_time(data: pd.DataFrame, time_col: str, target_col: str):
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
    df = data.sort_values(time_col)
    
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
        ax.xaxis.set_major_locator(HourLocator(interval=12))  # Mostrar ticks cada 6 horas
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
def single_normality_test(data: pd.DataFrame, variable: str) -> None:
    """
    Perform a normality test and plot a Q-Q plot for a single variable.

    Args:
        data: DataFrame
        variable: str (name of the feature to test and plot)
    
    Returns:
        None
    """
    display(Latex('Si el $pvalue$ < 0.05; se rechaza la $H_0$ sugiere que los datos no se ajustan de manera significativa a una distribución normal'))

    # Configurar la figura
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.suptitle(f"Prueba de Normalidad para '{variable}'", fontsize=12, y=1.05)
    
    # Gráfico Q-Q
    stats.probplot(data[variable], dist='norm', plot=ax)
    ax.set_xlabel(variable)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels())
    ax.grid(color='white', linestyle='-', linewidth=0.25)
    
    # Calcular p-value
    p_value = stats.normaltest(data[variable])[1]
    ax.text(0.05, 0.95, f"p-value = {p_value:.3f}", transform=ax.transAxes, fontsize=13, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
        
        
# Función para graficar una sola variable categórica
def single_categorical_plot(data: pd.DataFrame, variable: str) -> None:
    """
    Function to create a distribution graphic for a single categorical or discrete variable.

    Args:
        data: DataFrame
        variable: str (name of the feature to plot)
    
    Returns:
        None
    """
    # Crear una figura para el gráfico
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calcular los porcentajes de cada categoría
    temp_dataframe = pd.Series(data[variable].value_counts(normalize=True))
    temp_dataframe.sort_values(ascending=False).plot.bar(
        color='#f19900', edgecolor='skyblue', ax=ax
    )
    
    # Añadir una línea horizontal al 5% para resaltar las categorías poco comunes
    ax.axhline(y=0.05, color='#E51A4C', ls='dashed', lw=1.5)
    ax.set_ylabel('Porcentajes')
    ax.set_xlabel(variable)
    ax.set_xticklabels(temp_dataframe.index, rotation=25)
    ax.grid(color='white', linestyle='-', linewidth=0.25)
    
    # Añadir título
    plt.title(f"Categorical Plot for '{variable}'", fontsize=12, y=1.05)
    
    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()


# Función para graficar las categóricas segmentadas por el target
def categoricals_hue_target(data:pd.DataFrame, variables:list, target:str) -> None:
    
    # Graficos de cómo covarian algunas variables con respecto al target
    paletas = ['rocket', 'mako', 'crest', 'magma', 'viridis', 'flare']
    np.random.seed(11)

    for var in data[variables]:
        plt.figure(figsize=(11, 6))
        plt.title(f'{var} segmentado por {target}\n', fontsize=12)
        sns.countplot(x=var, hue=target, data=data, edgecolor='white', lw=0.5, palette=np.random.choice(paletas))
        plt.ylabel('Cantidades')
        plt.xticks(fontsize=12, rotation=25)
        plt.yticks(fontsize=12)
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        plt.tight_layout()