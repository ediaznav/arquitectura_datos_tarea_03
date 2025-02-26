### Funciones auxiliares en el código para el forecast 


## Funciones de Procesamiento de Datos -----------------------------------------------------

def proc_completness_check(df: pd.DataFrame, col1: str, col_id: str) -> pd.DataFrame:
    """
    Calcula y visualiza una tabla de completitud que muestra el porcentaje de cada valor
    de la columna pivot (col1) por cada grupo definido en la columna identificadora (col_id).

    Params:
        df (pd.DataFrame): DataFrame que contiene los datos.
        col1 (str): Nombre de la columna pivot para calcular los porcentajes.
        col_id (str): Nombre de la columna que se usa para agrupar los datos.

    Return:
        pd.DataFrame: Tabla con el porcentaje de cada valor de 'col1' por cada grupo en 'col_id'.
    """
    # Calcular la tabla de contingencia normalizada por fila y convertirla a porcentajes
    completeness_table = (df[[col1, col_id]].drop_duplicates().pivot_table(index = col_id, 
                              values = col1, 
                              aggfunc = 'count')/len(df[col1].drop_duplicates())).reset_index()


    

    return completeness_table

def proc_expand_dataframe(df: pd.DataFrame, new_col_name: str, values_list: list) -> pd.DataFrame:
    """
    Expande un DataFrame de pandas creando copias del DataFrame original, 
    donde cada copia se le añade una nueva columna con un valor fijo tomado 
    de 'values_list'. Todas las copias se apilan verticalmente.

    Parámetros:
        df (pd.DataFrame): DataFrame original.
        new_col_name (str): Nombre de la nueva columna a agregar.
        values_list (list): Lista de valores fijos para asignar a la nueva columna 
                            en cada copia.

    Retorna:
        pd.DataFrame: DataFrame expandido con todas las copias apiladas.
    """
    df_list = []
    
    for value in values_list:
        # Copiar el DataFrame original
        df_copy = df.copy()
        # Agregar la nueva columna con el valor fijo
        df_copy[new_col_name] = value
        df_list.append(df_copy)
    
    # Concatenar todas las copias en un solo DataFrame
    expanded_df = pd.concat(df_list, ignore_index=True)
    
    return expanded_df


## Funciones de Predicción de Datos -----------------------------------------------------


def pred_dividir_datos(serie, tamano_entrenamiento=26):
    """
    Divide una serie temporal en conjuntos de entrenamiento y prueba.

    Args:
        serie (list or np.array): Serie temporal completa.
        tamano_entrenamiento (int): Número de puntos para entrenamiento.

    Returns:
        tuple: (entrenamiento, prueba) como arrays de NumPy.
    """
    entrenamiento = np.array(serie[:tamano_entrenamiento])
    prueba = np.array(serie[tamano_entrenamiento:])
    return entrenamiento, prueba

def pred_ajustar_modelo(datos_entrenamiento, periodo_estacional=12):
    """
    Ajusta un modelo Holt-Winters sin tendencia y con estacionalidad.

    Args:
        datos_entrenamiento (np.array): Datos de entrenamiento.
        periodo_estacional (int): Período de la estacionalidad (12 en este caso).

    Returns:
        modelo: Modelo ajustado de Holt-Winters.
    """
    modelo = ExponentialSmoothing(
        datos_entrenamiento,
        seasonal_periods=periodo_estacional,
        trend=None,  # Sin tendencia
        seasonal='add'  # Estacionalidad aditiva
    ).fit(optimized=True)  # Optimiza automáticamente alpha, gamma
    return modelo


def pred_evaluar_prediccion(modelo, datos_prueba, pasos=6):
    """
    Realiza una predicción y evalúa el modelo con métricas MAE y RMSE.

    Args:
        modelo: Modelo Holt-Winters ajustado.
        datos_prueba (np.array): Datos reales para comparación.
        pasos (int): Número de pasos a predecir.

    Returns:
        dict: Métricas de evaluación y predicciones.
    """
    prediccion = modelo.forecast(pasos)
    mae = mean_absolute_error(datos_prueba, prediccion)
    rmse = np.sqrt(mean_squared_error(datos_prueba, prediccion))
    
    return {
        'prediccion': prediccion,
        'MAE': mae,
        'RMSE': rmse
    }

def pred_prediccion_final(serie_completa, periodo_estacional=12, pasos=6):
    """
    Ajusta el modelo a toda la serie y predice los siguientes pasos.

    Args:
        serie_completa (list or np.array): Serie temporal completa.
        periodo_estacional (int): Período de la estacionalidad.
        pasos (int): Número de pasos a predecir.

    Returns:
        np.array: Predicciones futuras.
    """
    modelo_final = ExponentialSmoothing(
        np.array(serie_completa),
        seasonal_periods=periodo_estacional,
        trend=None,
        seasonal='add'
    ).fit(optimized=True)
    prediccion_futura = modelo_final.forecast(pasos)
    return prediccion_futura

def pred_next_n_months(period: str, n: int) -> list:
    """
    Dado un período en formato 'yyyy-mm' y un número entero n, retorna una lista
    con los siguientes n meses en el mismo formato.

    Parámetros:
        period (str): Período inicial en formato 'yyyy-mm'.
        n (int): Número de meses siguientes a generar.

    Retorna:
        list: Lista de cadenas de texto con los períodos en formato 'yyyy-mm'.
    """
    # Convertir el período de string a un objeto datetime
    try:
        current_date = datetime.datetime.strptime(period, "%Y-%m")
    except ValueError:
        raise ValueError("El período debe estar en formato 'yyyy-mm'")

    months_list = []
    
    # Generar los siguientes n meses
    for _ in range(n):
        # Si el mes es diciembre, reinicia a enero y suma un año;
        # de lo contrario, solo incrementa el mes
        if current_date.month == 12:
            next_year = current_date.year + 1
            next_month = 1
        else:
            next_year = current_date.year
            next_month = current_date.month + 1

        # Crear un nuevo objeto datetime para el siguiente mes (día 1)
        current_date = datetime.datetime(next_year, next_month, 1)
        # Agregar el período formateado a la lista
        months_list.append(current_date.strftime("%Y-%m"))

    return months_list


def pred_graficar_ventas(df, output_dir="../reports/figures/", filename="ventas.png"):
    """
    Función para graficar las ventas mensuales por tienda, comparando la predicción con los valores reales,
    y guardar la gráfica en un directorio especificado.

    Params:
    df (pd.DataFrame): DataFrame que contiene las predicciones y los valores reales.
    output_dir (str): Directorio donde se guardará la imagen. Default "graficos".
    filename (str): Nombre del archivo de salida. Default "ventas.png".

    Return:
    None: Guarda la gráfica en el directorio especificado.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(10, 6))
    plt.title("Ventas mensuales por tienda \nReal contra predicción.")

    # Graficar todas las columnas en gris para la predicción
    for c in df.columns.tolist():
        plt.plot(df[c], alpha=0.7, color='gray', linestyle='-.')
    
    plt.plot(df[c], alpha=0.7, color='gray', label='Predicción')
    
    # Graficar la serie real en azul oscuro
    plt.plot(df.reset_index().drop(columns=['period']), color='darkblue')
    plt.plot(df.reset_index().drop(columns=['period'])[c], color='darkblue', label='Real')

    plt.xlabel("Número de periodo")
    plt.ylabel("Ventas mensuales por tienda")
    plt.legend()
    
    # Guardar la gráfica en el archivo especificado
    plt.savefig(output_path)
    plt.close()









