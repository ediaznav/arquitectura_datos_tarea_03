### Importamos librerias necesarias

import pandas as pd
import numpy as np
from utils import *


### Parametros y configuraciones 

## Parametros 
DATA_RAW_DIRECTORY = "../data/raw/"
DATA_PREP_DIRECTORY = "../data/prep/"
ITEM_MONTHLY_SALE_LIMIT = 15
ITEM_MONTHLY_AVGPRICE_LIMIT = 6000

## Configuraciones 


### PROCESO DE PREPARACIÓN DE DATOS 

# Importamos los datos a explorar

item_categories = pd.read_csv(DATA_RAW_DIRECTORY + "item_categories.csv")
items = pd.read_csv(DATA_RAW_DIRECTORY +"items.csv")
sales_train = pd.read_csv(DATA_RAW_DIRECTORY +"sales_train.csv")
shops = pd.read_csv(DATA_RAW_DIRECTORY +"shops.csv")
test = pd.read_csv(DATA_RAW_DIRECTORY +"test.csv")
sample_submission = pd.read_csv(DATA_RAW_DIRECTORY + "sample_submission.csv")


# Formato y manejo de fechas 
sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')
sales_train['period'] = sales_train['date'].dt.to_period("M")
sales_train['month'] = sales_train['date'].dt.to_period("M").dt.strftime("%m")
sales_train['year'] = sales_train['date'].dt.to_period("Y").dt.strftime("%Y")

# Eliminamos valore negativos, nulos o ceros.  
sales_train = sales_train[sales_train.item_cnt_day >= 0].dropna(subset = ['item_cnt_day'])
sales_train = sales_train[sales_train.item_price >= 0].dropna(subset = ['item_price'])

# Se agrupa a nivel mensual
sales_month_train = sales_train.pivot_table(index = ['period','year' ,'month','date_block_num','shop_id', 'item_id'],
                                            values = 'item_cnt_day', aggfunc = np.sum
                                            ).rename(columns = {'item_cnt_day':'item_cnt_month'}).reset_index().merge(
                                            sales_train.pivot_table(index = ['period','year' ,'month','date_block_num','shop_id', 'item_id'],
                                            values = 'item_price', aggfunc = np.mean).rename(columns = {'item_price':'item_price_mavg'}).reset_index(), 
                                            on = ['period','year' ,'month','date_block_num','shop_id', 'item_id'], 
                                            how = 'left'
                                            )


complete_shops = proc_completness_check(sales_month_train, 'period', 'shop_id')
complete_shops_list = complete_shops[complete_shops.period == 1].shop_id.drop_duplicates().tolist()
complete_items = proc_completness_check(sales_month_train, 'period', 'item_id')
complete_items_list = complete_items[complete_items.period == 1].item_id.drop_duplicates().tolist()

items_shops_catalog = sales_month_train[sales_month_train.shop_id.isin(complete_shops_list)][['shop_id', 'item_id']].drop_duplicates()


# Seleccionamos la data valida 

# Tiendas a realizar el forecast 
sales_month_train_not_complete = sales_month_train[~sales_month_train.shop_id.isin(complete_shops_list)]
sales_month_train = sales_month_train[sales_month_train.shop_id.isin(complete_shops_list)]

# Solo valores positivos en compras 
sales_month_train = sales_month_train[sales_month_train.item_cnt_month >= 0]
sales_month_train = sales_month_train[sales_month_train.item_price_mavg >= 0]
sales_month_train = sales_month_train[sales_month_train.item_cnt_month < ITEM_MONTHLY_SALE_LIMIT]
sales_month_train = sales_month_train[sales_month_train.item_price_mavg < ITEM_MONTHLY_AVGPRICE_LIMIT]



# Columnas no usadas
sales_month_train.drop(columns = ['date_block_num'], inplace = True)

# Data agregada por tienda 

sales_month_shop_train = sales_month_train.pivot_table(
    index = 'period', 
    columns = 'shop_id', 
    values = 'item_cnt_month', 
    aggfunc = np.sum
).astype("int")


# Generamos el share de las ventas por item y shop 

last_month_sales = items_shops_catalog.merge(sales_month_train[sales_month_train.period == last_period][['shop_id', 'item_id', 'item_cnt_month']], 
    on = ['shop_id', 'item_id'], 
    how = 'left'
).fillna(0)

last_month_sales = last_month_sales.merge(
    last_month_sales.groupby("shop_id").sum()[['item_cnt_month']].reset_index().rename(columns = {'item_cnt_month':'month_sales'}), 
    on = 'shop_id', 
    how = 'left'
)

last_month_sales['item_share'] = last_month_sales['item_cnt_month']/last_month_sales['month_sales']


### Proceso de Exportación de Datos preparados 

sales_train.to_csv(DATA_PREP_DIRECTORY + "sales_train.csv", index = False)
complete_shops.to_csv(DATA_PREP_DIRECTORY + "complete_shops.csv", index = False)
complete_items.to_csv(DATA_PREP_DIRECTORY + "complete_items.csv", index = False)
items_shops_catalog.to_csv(DATA_PREP_DIRECTORY + "items_shops_catalog.csv", index = False)
sales_month_train_not_complete.to_csv(DATA_PREP_DIRECTORY + "sales_month_train_not_complete.csv", index = False)
sales_month_shop_train.to_csv(DATA_PREP_DIRECTORY + "sales_month_shop_train.csv", index = True)
last_month_sales.to_csv(DATA_PREP_DIRECTORY + "last_month_sales.csv", index = False)
