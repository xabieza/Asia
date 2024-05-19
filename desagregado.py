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

def pivotadosk(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['Country ID', 'Continent', 'ISO 3', 'Continent ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Filtrar elementos de la columna 'Country' que queremos mantener
    elementos_a_mantener = ['Hong Kong','Japan','Austria','India','New Zealand','China','Bangladesh','Indonesia','Thailand','Singapore','Vietnam']  # Reemplaza estos valores con los que necesites
    df_filtered = mantener_elementos(df_cleaned, 'Country', elementos_a_mantener)
    
    # Agrupar por Country y Year, sumando Trade Value
    df_final = df_filtered.groupby(['Country', 'Year']).sum().reset_index()
    
    # Pivotar el DataFrame
    #pivoted_df = df_final.pivot(index='Country', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return df_final

#singapure
def pivotadosp(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['Country ID', 'Continent', 'ISO 3', 'Continent ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Filtrar elementos de la columna 'Country' que queremos mantener
    elementos_a_mantener = ['Hong Kong','Japan','Austria','China','New Zealand','India','Bangladesh','Indonesia','Thailand','South Korea','Vietnam']  # Reemplaza estos valores con los que necesites
    df_filtered = mantener_elementos(df_cleaned, 'Country', elementos_a_mantener)
    
    # Agrupar por Country y Year, sumando Trade Value
    df_final = df_filtered.groupby(['Country', 'Year']).sum().reset_index()
    
    # Pivotar el DataFrame
    #pivoted_df = df_final.pivot(index='Country', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return df_final
#China
def pivotadoch(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['Country ID', 'Continent', 'ISO 3', 'Continent ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Filtrar elementos de la columna 'Country' que queremos mantener
    elementos_a_mantener = ['Hong Kong','Japan','Austria','New Zealand','Singapore','India','Bangladesh','Indonesia','Thailand','South Korea','Vietnam']  # Reemplaza estos valores con los que necesites
    df_filtered = mantener_elementos(df_cleaned, 'Country', elementos_a_mantener)
    
    # Agrupar por Country y Year, sumando Trade Value
    df_final = df_filtered.groupby(['Country', 'Year']).sum().reset_index()
    
    # Pivotar el DataFrame
    #pivoted_df = df_final.pivot(index='Country', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return df_final

#Hong kong
def pivotadoHK(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['Country ID', 'Continent', 'ISO 3', 'Continent ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Filtrar elementos de la columna 'Country' que queremos mantener
    elementos_a_mantener = ['China','Japan','Austria','New Zealand','Singapore','India','Bangladesh','Indonesia','Thailand','South Korea','Vietnam']  # Reemplaza estos valores con los que necesites
    df_filtered = mantener_elementos(df_cleaned, 'Country', elementos_a_mantener)
    
    # Agrupar por Country y Year, sumando Trade Value
    df_final = df_filtered.groupby(['Country', 'Year']).sum().reset_index()
    
    # Pivotar el DataFrame
    #pivoted_df = df_final.pivot(index='Country', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return df_final

#India
def pivotadoind(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['Country ID', 'Continent', 'ISO 3', 'Continent ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Filtrar elementos de la columna 'Country' que queremos mantener
    elementos_a_mantener = ['China','Japan','Austria','New Zealand','Singapore','Hong Kong','Bangladesh','Indonesia','Thailand','South Korea','Vietnam']  # Reemplaza estos valores con los que necesites
    df_filtered = mantener_elementos(df_cleaned, 'Country', elementos_a_mantener)
    
    # Agrupar por Country y Year, sumando Trade Value
    df_final = df_filtered.groupby(['Country', 'Year']).sum().reset_index()
    
    # Pivotar el DataFrame
    #pivoted_df = df_final.pivot(index='Country', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return df_final

#vietnam
def pivotadovt(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['Country ID', 'Continent', 'ISO 3', 'Continent ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Filtrar elementos de la columna 'Country' que queremos mantener
    elementos_a_mantener = ['China','Japan','Austria','New Zealand','Singapore','Hong Kong','Bangladesh','Indonesia','Thailand','South Korea','India']  # Reemplaza estos valores con los que necesites
    df_filtered = mantener_elementos(df_cleaned, 'Country', elementos_a_mantener)
    
    # Agrupar por Country y Year, sumando Trade Value
    df_final = df_filtered.groupby(['Country', 'Year']).sum().reset_index()
    
    # Pivotar el DataFrame
    #pivoted_df = df_final.pivot(index='Country', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return df_final

#Bangladesh
#vietnam
def pivotadoBG(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['Country ID', 'Continent', 'ISO 3', 'Continent ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Filtrar elementos de la columna 'Country' que queremos mantener
    elementos_a_mantener = ['China','Japan','Austria','New Zealand','Singapore','Hong Kong','Vietnam','Indonesia','Thailand','South Korea','India']  # Reemplaza estos valores con los que necesites
    df_filtered = mantener_elementos(df_cleaned, 'Country', elementos_a_mantener)
    
    # Agrupar por Country y Year, sumando Trade Value
    df_final = df_filtered.groupby(['Country', 'Year']).sum().reset_index()
    
    # Pivotar el DataFrame
    #pivoted_df = df_final.pivot(index='Country', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return df_final


###########################################################################################################################################################
#concatenar e IGL
def concat_and_rename_columns(df1, df2):
    # Renombrar las columnas del segundo DataFrame
    df2_renamed = df2.rename(columns={'Country': 'Country 2', 'Year': 'Year 2', 'Trade Value': 'Trade Value 2'})

    # Unir los DataFrames utilizando merge
    result = pd.merge(df1, df2_renamed, left_on=['Country', 'Year'], right_on=['Country 2', 'Year 2'], how='inner')

    # Eliminar las columnas duplicadas (Country 2 y Year 2)
    result = result.drop(['Country 2', 'Year 2'], axis=1)

    # Calcular la diferencia en valor absoluto entre Trade Value y Trade Value 2
    result['Diferencia'] = abs(result['Trade Value'] - result['Trade Value 2'])

    # Crear una nueva columna que sume los valores de Trade Value y Trade Value 2
    result['Suma'] = result['Trade Value'] + result['Trade Value 2']

    # Calcular la columna IGL (Índice de Ganancia o Pérdida)
    result['IGL'] = 1- (result['Diferencia'] / result['Suma'])

    # Eliminar las columnas Trade Value, Trade Value 2, Diferencia y Suma
    result = result.drop(['Trade Value', 'Trade Value 2', 'Diferencia', 'Suma'], axis=1)

    # Realizar la operación de pivote con Year como columnas y Country como índice
    pivoted_result = result.pivot_table(index='Country', columns='Year', values='IGL', fill_value=0).rename_axis(columns=None).reset_index()

    return pivoted_result

####################################################################################################################################################
#####################################################################################################

# Hacer una media de cada año

def manteneraño(df):
    # Seleccionar las columnas 'Country' y '2022'
    result = df[['Country', 2022]]
    return result
###########################################################################################

##### Concatenar dataframe al mismo nivel.

def organizar_tabla(df_south_korea, df_singapore, df_china, df_hongkong, df_india, df_vietnam, df_bg):
    # Crear un nuevo DataFrame con los países como índice y rellenar con los datos correspondientes
    df = pd.DataFrame(columns=['South Korea', 'Singapore', 'China','Hong Kong','India','Vietnam','Bangladesh'], index=df_south_korea['Country'])

    # Llenar el DataFrame con los valores correspondientes de los DataFrames originales
    for country in df_south_korea['Country']:
        df.loc[country, 'South Korea'] = df_south_korea.loc[df_south_korea['Country'] == country, 2022].values[0]

    for country in df_singapore['Country']:
        df.loc[country, 'Singapore'] = df_singapore.loc[df_singapore['Country'] == country, 2022].values[0]

    for country in df_china['Country']:
        df.loc[country, 'China'] = df_china.loc[df_china['Country'] == country, 2022].values[0]
    
    for country in df_hongkong['Country']:
        df.loc[country, 'Hong Kong'] = df_hongkong.loc[df_hongkong['Country'] == country, 2022].values[0]
        
    for country in df_india['Country']:
        df.loc[country, 'India'] = df_india.loc[df_india['Country'] == country, 2022].values[0]
    
    for country in df_vietnam['Country']:
        df.loc[country, 'Vietnam'] = df_vietnam.loc[df_vietnam['Country'] == country, 2022].values[0]
    
    for country in df_bg['Country']:
        df.loc[country, 'Bangladesh'] = df_bg.loc[df_bg['Country'] == country, 2022].values[0]

    # Resetear el índice para mover 'Country' como una columna al mismo nivel que las otras columnas
    df = df.reset_index()

    return df

#########################################




#para el resto del mundo


### Para limpiar los dataframe de mierda y tenerlos pivotados.
def mantener_elementos(df, Country, elementos_a_mantener):
     #Filtrar el DataFrame para mantener solo los elementos deseados en la columna especificada
     df_filtrado = df[df[Country].isin(elementos_a_mantener)]
     return df_filtrado

def pivotadoskout(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['Country ID', 'Continent', 'ISO 3', 'Continent ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Filtrar elementos de la columna 'Country' que queremos mantener
    elementos_a_mantener = ['Germany','Finland', 'Denmark', 'Spain', 'Estonia', 'Finland', 'France',
       'United Kingdom', 'Greece', 'Canada','United States', 'Hungary', 'Ireland','Mexico','Sweden']  # Reemplaza estos valores con los que necesites
    df_filtered = mantener_elementos(df_cleaned, 'Country', elementos_a_mantener)
    
    # Agrupar por Country y Year, sumando Trade Value
    df_final = df_filtered.groupby(['Country', 'Year']).sum().reset_index()
    
    # Pivotar el DataFrame
    #pivoted_df = df_final.pivot(index='Country', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return df_final

#singapure
def pivotadospout(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['Country ID', 'Continent', 'ISO 3', 'Continent ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Filtrar elementos de la columna 'Country' que queremos mantener
    elementos_a_mantener = ['Germany','Finland', 'Denmark', 'Spain', 'Estonia', 'Finland', 'France',
       'United Kingdom', 'Greece', 'Canada','United States', 'Hungary', 'Ireland','Mexico','Sweden']  # Reemplaza estos valores con los que necesites
    df_filtered = mantener_elementos(df_cleaned, 'Country', elementos_a_mantener)
    
    # Agrupar por Country y Year, sumando Trade Value
    df_final = df_filtered.groupby(['Country', 'Year']).sum().reset_index()
    
    # Pivotar el DataFrame
    #pivoted_df = df_final.pivot(index='Country', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return df_final
#China
def pivotadochout(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['Country ID', 'Continent', 'ISO 3', 'Continent ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Filtrar elementos de la columna 'Country' que queremos mantener
    elementos_a_mantener = ['Germany','Finland', 'Denmark', 'Spain', 'Estonia', 'Finland', 'France',
       'United Kingdom', 'Greece', 'Canada','United States', 'Hungary', 'Ireland','Sweden','Mexico']  # Reemplaza estos valores con los que necesites
    df_filtered = mantener_elementos(df_cleaned, 'Country', elementos_a_mantener)
    
    # Agrupar por Country y Year, sumando Trade Value
    df_final = df_filtered.groupby(['Country', 'Year']).sum().reset_index()
    
    # Pivotar el DataFrame
    #pivoted_df = df_final.pivot(index='Country', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return df_final

# Bangladesh
def pivotadobgout(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['Country ID', 'Continent', 'ISO 3', 'Continent ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Filtrar elementos de la columna 'Country' que queremos mantener
    elementos_a_mantener = ['Germany','Finland', 'Denmark', 'Spain', 'Estonia', 'Finland', 'France',
       'United Kingdom', 'Greece', 'Canada','United States', 'Hungary', 'Ireland','Sweden','Mexico']  # Reemplaza estos valores con los que necesites
    df_filtered = mantener_elementos(df_cleaned, 'Country', elementos_a_mantener)
    
    # Agrupar por Country y Year, sumando Trade Value
    df_final = df_filtered.groupby(['Country', 'Year']).sum().reset_index()
    
    # Pivotar el DataFrame
    #pivoted_df = df_final.pivot(index='Country', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return df_final

