
### Importamos las librerias Necesarias 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import warnings
import datetime
from utils import *


### Parametros y configuraciones 

## Parametros 
DATA_PREP_DIRECTORY = "../data/prep/"
DATA_PREDICTION_DIRECTORY = "../data/predictions/"
FIGURES_DIRECTORY = "../reports/figures/"
PREDICTION_PERIOD_WINDOW = 6


## Configuraciones

# Estilo de matplotlib
plt.style.use("ggplot")



### Predicciones del Modelo

## Importamos los datos necesarios 

# Base de datos preparada para el ajuste 
sales_month_shop_train = pd.read_csv(DATA_PREP_DIRECTORY + "sales_month_shop_train.csv")
last_month_sales = pd.read_csv(DATA_PREP_DIRECTORY + "last_month_sales.csv")

## Ajustamos y predecimosel modelo Holt-Winters 

# Agrupar los resultados
forecasts = pd.DataFrame()
for c in sales_month_shop_train.columns.tolist():
    serie = sales_month_shop_train[c]
    prediccion_futura = pred_prediccion_final(serie)    
    forecasts[c] = prediccion_futura

# Redondeamos los estimados a enteros 
forecasts = forecasts.round().astype("int")

# Obtenemos el último periodo de la base histórica 
last_period = sales_month_shop_train.index.max().strftime("%Y-%m")

# Actualizamos los indices en nuestra tabla de predicciones 
forecasts.index = pred_next_n_months(last_period,PREDICTION_PERIOD_WINDOW)

# Re index el dataFrame forecasts 
forecasts = forecasts.reset_index().rename(columns = {'index':'period'})

# Guardamos una grafica de los historicos contra el forecast por tienda 
pred_graficar_ventas(sales_month_shop_train, 'item_cnt_month', output_dir= FIGURES_DIRECTORY , 
                     filename="ventas_reales_contra_forecast.png")

# Derretimos las predicciones en una tabla 
forecasts_melt = forecasts.melt(
    id_vars='period', 
    var_name = 'shop_id', 
    value_name= 'month_sales'
)

# Unimos forecast con shares 
forecasts_items = last_month_sales[['shop_id', 'item_id', 'item_share']].merge(forecasts_melt, on = 'shop_id', how = 'left')
forecasts_items['item_cnt_month'] = (forecasts_items['month_sales']*forecasts_items['item_share']).round().astype("int")


# Generamos la tabla final 
forecasts_final_df = forecasts_items.copy()

# forecast 
forecasts_final_df.to_csv(DATA_PREDICTION_DIRECTORY +
                           f"sales_prediction_{datetime.today().strftime('%Y-%m-%d')}_{PREDICTION_PERIOD_WINDOW}.csv", 
                           index = False)
