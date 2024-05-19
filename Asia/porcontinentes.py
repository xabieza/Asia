import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import seaborn as sns


### Para limpiar los dataframe de mierda y tenerlos pivotados.
def mantener_elementos(df, Country, elementos_a_mantener):
     #Filtrar el DataFrame para mantener solo los elementos deseados en la columna especificada
     df_filtrado = df[df[Country].isin(elementos_a_mantener)]
     return df_filtrado

def pivotadoscontinent(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['Country ID', 'Country', 'ISO 3', 'Continent ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    df_final = df_cleaned.groupby(['Continent', 'Year']).sum().reset_index()
    

    
    return df_final

###########################################################################################################################################################
#concatenar e IGL
def concat_and_rename_columnscont(df1, df2):
    # Renombrar las columnas del segundo DataFrame
    df2_renamed = df2.rename(columns={'Continent': 'Continent 2', 'Year': 'Year 2', 'Trade Value': 'Trade Value 2'})

    # Unir los DataFrames utilizando merge
    result = pd.merge(df1, df2_renamed, left_on=['Continent', 'Year'], right_on=['Continent 2', 'Year 2'], how='inner')

    # Eliminar las columnas duplicadas (Country 2 y Year 2)
    result = result.drop(['Continent 2', 'Year 2'], axis=1)

    # Calcular la diferencia en valor absoluto entre Trade Value y Trade Value 2
    result['Diferencia'] = abs(result['Trade Value'] - result['Trade Value 2'])

    # Crear una nueva columna que sume los valores de Trade Value y Trade Value 2
    result['Suma'] = result['Trade Value'] + result['Trade Value 2']

    # Calcular la columna IGL (Índice de Ganancia o Pérdida)
    result['IGL'] = 1-(result['Diferencia'] / result['Suma'])

    # Eliminar las columnas Trade Value, Trade Value 2, Diferencia y Suma
    result = result.drop(['Trade Value', 'Trade Value 2', 'Diferencia', 'Suma'], axis=1)

    # Realizar la operación de pivote con Year como columnas y Country como índice
    pivoted_result = result.pivot_table(index='Continent', columns='Year', values='IGL', fill_value=0).rename_axis(columns=None).reset_index()

    return pivoted_result

####################################################################################################################################################
#####################################################################################################

# Hacer una media de cada año

def mantenerañoct(df):
    # Seleccionar las columnas 'Country' y '2022'
    result = df[['Continent', 2022]]
    return result

###########################################################################################

##### Concatenar dataframe al mismo nivel.

def organizar_tablacont(df_south_korea, df_singapore, df_china,df_hk,df_ind,df_viet,df_bg):
    # Crear un nuevo DataFrame con los países como índice y rellenar con los datos correspondientes
    df = pd.DataFrame(columns=['South Korea', 'Singapore', 'China'], index=df_south_korea['Continent'])

    # Llenar el DataFrame con los valores correspondientes de los DataFrames originales
    for Continent in df_south_korea['Continent']:
        df.loc[Continent, 'South Korea'] = df_south_korea.loc[df_south_korea['Continent'] == Continent, 2022].values[0]

    for Continent in df_singapore['Continent']:
        df.loc[Continent, 'Singapore'] = df_singapore.loc[df_singapore['Continent'] == Continent, 2022].values[0]

    for Continent in df_china['Continent']:
        df.loc[Continent, 'China'] = df_china.loc[df_china['Continent'] == Continent, 2022].values[0]
    
    for Continent in df_hk['Continent']:
        df.loc[Continent, 'hong Kong'] = df_hk.loc[df_hk['Continent'] == Continent, 2022].values[0]

    for Continent in df_ind['Continent']:
        df.loc[Continent, 'India'] = df_ind.loc[df_ind['Continent'] == Continent, 2022].values[0]
  
    for Continent in df_viet['Continent']:
        df.loc[Continent, 'Vietnam'] = df_viet.loc[df_viet['Continent'] == Continent, 2022].values[0]
  
    for Continent in df_bg['Continent']:
        df.loc[Continent, 'bangladesh'] = df_bg.loc[df_bg['Continent'] == Continent, 2022].values[0] 
    # Resetear el índice para mover 'Country' como una columna al mismo nivel que las otras columnas
    df = df.reset_index()

    return df

#########################################



