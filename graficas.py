import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import seaborn as sns


# Sin peso relativo ni nada con el bruto

### Para limpiar los dataframe de mierda y tenerlos pivotados.
def mantener_elementos(df, Country, elementos_a_mantener):
     #Filtrar el DataFrame para mantener solo los elementos deseados en la columna especificada
     df_filtrado = df[df[Country].isin(elementos_a_mantener)]
     return df_filtrado

def sinporcentaje(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['HS2', 'HS2 ID', 'HS4', 'HS4 ID', 'Section ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Agrupar por Country y Year, sumando Trade Value
    df_final = df_cleaned.groupby(['Section', 'Year']).sum().reset_index()
    
    # Pivotar el DataFrame
    pivoted_df = df_final.pivot(index='Section', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return pivoted_df

#########################################################################################################



### Limpiar la tabla y no pivota.

def mantener_elementos(df, Country, elementos_a_mantener):
     #Filtrar el DataFrame para mantener solo los elementos deseados en la columna especificada
     df_filtrado = df[df[Country].isin(elementos_a_mantener)]
     return df_filtrado

def limpia(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['HS2', 'HS2 ID', 'HS4', 'HS4 ID', 'Section ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Agrupar por Country y Year, sumando Trade Value
    df_final = df_cleaned.groupby(['Section', 'Year']).sum().reset_index()
    
    # Pivotar el DataFrame
    #pivoted_df = df_final.pivot(index='Section', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return df_final

#####################################################
#  con el porcentaje relativo
def calcular_porcentaje_peso(df):
    # Calcula la suma del Trade Value para cada sección por año
    df['Total_por_Año'] = df.groupby('Year')['Trade Value'].transform('sum')
    
    # Calcula el porcentaje de peso de cada sección en cada año
    df['Porcentaje_de_Peso'] = (df['Trade Value'] / df['Total_por_Año']) * 100
    
    # Crea un nuevo DataFrame con el porcentaje de peso de cada sección en cada año
    nuevo_df = df[['Section', 'Year', 'Porcentaje_de_Peso']].copy()
    pivoted_df = nuevo_df.pivot(index='Section', columns='Year', values='Porcentaje_de_Peso').rename_axis(columns=None).reset_index()
    return pivoted_df

############################################################################################################
###################################################################################################


# para quitar esos años indeseados
def años_productos(df):
    # Elementos no deseados en 'Section'
    valores_no_deseados = ['Metals', 'Mineral Products', 'Miscellaneous', 'Paper Goods',
                           'Plastics and Rubbers', 'Precious Metals', 'Wood Products',
                           'Weapons', 'Instruments', 'Stone And Glass']

    # Columnas no deseadas
    valores_no_deseados2 = [
        1996, 1997, 1998, 1999,
        2000, 2001, 2002, 2003, 2004,
        2006, 2007, 2008, 2009,
        2011, 2012, 2013, 2014,
        2016, 2017, 2018, 2019
    ]

    # Eliminar filas basadas en elementos no deseados en 'Section'
    df = df[~df['Section'].isin(valores_no_deseados)]

    # Eliminar columnas basadas en valores no deseados
    df = df.drop(columns=valores_no_deseados2, errors='ignore')

    return df
##########################################################################################
##############################################################################################################################

#Función para realizar la concatenación y sumar las columnas 'Trade Value'
def volumentotal(df1, df2):
    # Renombrar las columnas Trade Value
    df1.rename(columns={'Trade Value': 'Exportaciones'}, inplace=True)
    df2.rename(columns={'Trade Value': 'Importaciones'}, inplace=True)

    # Fusionar los DataFrames en base a las columnas 'Section' y 'Year'
    df_merged = pd.merge(df1, df2, on=['Section', 'Year'], how='outer')

    # Llenar los valores NaN con 0 para la suma
    df_merged['Exportaciones'].fillna(0, inplace=True)
    df_merged['Importaciones'].fillna(0, inplace=True)

    # Calcular la columna 'Volumen Total'
    df_merged['Volumen Total'] = df_merged['Exportaciones'] + df_merged['Importaciones']
    df_merged['Diferencia'] = df_merged['Exportaciones'] - df_merged['Importaciones']
    
    # Seleccionar las columnas requeridas
    df_final = df_merged[['Year','Section','Volumen Total']]
    
    #pivoted_df = df_final.pivot(index='Section', columns='Year', values='Volumen Total').rename_axis(columns=None).reset_index()
    return df_final 

#Tabla en pormenorizado 
def exportacioneseimportaciones(df1, df2):
    # Renombrar las columnas Trade Value
    df1.rename(columns={'Trade Value': 'Exportaciones'}, inplace=True)
    df2.rename(columns={'Trade Value': 'Importaciones'}, inplace=True)

    # Fusionar los DataFrames en base a las columnas 'Section' y 'Year'
    df_merged = pd.merge(df1, df2, on=['Section', 'Year'], how='outer')

    # Llenar los valores NaN con 0 para la suma
    df_merged['Exportaciones'].fillna(0, inplace=True)
    df_merged['Importaciones'].fillna(0, inplace=True)

    # Calcular la columna 'Volumen Total'
    df_merged['Volumen Total'] = df_merged['Exportaciones'] + df_merged['Importaciones']
    df_merged['Diferencia'] = df_merged['Exportaciones'] - df_merged['Importaciones']
    # Seleccionar las columnas requeridas
    df_final = df_merged[['Year','Section', 'Exportaciones', 'Importaciones','Volumen Total','Diferencia']]

    return df_final
#  con el porcentaje relativo
def calcular_porcentaje_pesotablafinal(df):
    # Calcula la suma del Trade Value para cada sección por año
    df['Total_por_Año'] = df.groupby('Year')['Volumen Total'].transform('sum')
    
    # Calcula el porcentaje de peso de cada sección en cada año
    df['Porcentaje_de_Peso'] = (df['Volumen Total'] / df['Total_por_Año']) * 100
    
    # Crea un nuevo DataFrame con el porcentaje de peso de cada sección en cada año
    nuevo_df = df[['Section', 'Year', 'Porcentaje_de_Peso']].copy()
    pivoted_df = nuevo_df.pivot(index='Section', columns='Year', values='Porcentaje_de_Peso').rename_axis(columns=None).reset_index()
    return pivoted_df




###############################################################################################
####################################################################################################
###############################################################################################

#CONTINENTES

### Limpiar la tabla y no pivota.

def mantener_elementos(df, Country, elementos_a_mantener):
     #Filtrar el DataFrame para mantener solo los elementos deseados en la columna especificada
     df_filtrado = df[df[Country].isin(elementos_a_mantener)]
     return df_filtrado

def limpiacont(df):
    # Eliminar columnas especificadas
    columns_to_drop = ['Continent ID', 'Country', 'Country ID', 'ISO 3']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Agrupar por Country y Year, sumando Trade Value
    df_final = df_cleaned.groupby(['Continent', 'Year']).sum().reset_index()
    
    # Pivotar el DataFrame
    #pivoted_df = df_final.pivot(index='Section', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return df_final

#############################################################################################################

### Para limpiar los dataframe de mierda y tenerlos pivotados.
def mantener_elementos(df, Country, elementos_a_mantener):
     df_filtrado = df[df[Country].isin(elementos_a_mantener)]
     return df_filtrado

def sinporcentajecont(df):
    columns_to_drop = ['Continent ID', 'Country', 'Country ID', 'ISO 3']
    df_cleaned = df.drop(columns=columns_to_drop)
    df_final = df_cleaned.groupby(['Continent', 'Year']).sum().reset_index()
    pivoted_df = df_final.pivot(index='Continent', columns='Year', values='Trade Value').rename_axis(columns=None).reset_index()
    
    return pivoted_df
##################################################################################
#  con el porcentaje relativo
def calcular_porcentaje_pesocnt(df):
    # Calcula la suma del Trade Value para cada sección por año
    df['Total_por_Año'] = df.groupby('Year')['Trade Value'].transform('sum')
    
    # Calcula el porcentaje de peso de cada sección en cada año
    df['Porcentaje_de_Peso'] = ((df['Trade Value'] / df['Total_por_Año'])) * 100
    
    # Crea un nuevo DataFrame con el porcentaje de peso de cada sección en cada año
    nuevo_df = df[['Continent', 'Year', 'Porcentaje_de_Peso']].copy()
    pivoted_df = nuevo_df.pivot(index='Continent', columns='Year', values='Porcentaje_de_Peso').rename_axis(columns=None).reset_index()
    return pivoted_df


#Función para realizar la concatenación y sumar las columnas 'Trade Value'
def volumentotalcont(df1, df2):

    df1.rename(columns={'Trade Value': 'Exportaciones'}, inplace=True)
    df2.rename(columns={'Trade Value': 'Importaciones'}, inplace=True)

   
    df_merged = pd.merge(df1, df2, on=['Continent', 'Year'], how='outer')

    df_merged['Exportaciones'].fillna(0, inplace=True)
    df_merged['Importaciones'].fillna(0, inplace=True)

    df_merged['Volumen Total'] = df_merged['Exportaciones'] + df_merged['Importaciones']
    df_merged['Diferencia'] = df_merged['Exportaciones'] - df_merged['Importaciones']
    
    df_final = df_merged[['Year','Continent','Volumen Total']]
    
    #pivoted_df = df_final.pivot(index='Continent', columns='Year', values='Volumen Total').rename_axis(columns=None).reset_index()
    return df_final
########################################################################################################################
##################################################################################################################################
## La tabla con todas las operaciones y el concatenado
def exportacioneseimportacionescont(df1, df2):
    
    df1.rename(columns={'Trade Value': 'Exportaciones'}, inplace=True)
    df2.rename(columns={'Trade Value': 'Importaciones'}, inplace=True)

    df_merged = pd.merge(df1, df2, on=['Continent', 'Year'], how='outer')

    df_merged['Exportaciones'].fillna(0, inplace=True)
    df_merged['Importaciones'].fillna(0, inplace=True)

    df_merged['Volumen Total'] = df_merged['Exportaciones'] + df_merged['Importaciones']
    df_merged['Diferencia'] = df_merged['Exportaciones'] - df_merged['Importaciones']
   
    df_final = df_merged[['Year','Continent', 'Exportaciones', 'Importaciones','Diferencia','Volumen Total']]

    return df_final

# para quitar esos años indeseados
def quitarañoscont(df):
    
    valores_no_deseados2 = [
        1996, 1997, 1998, 1999,
        2001, 2002, 2003, 2004,
        2006, 2007, 2008, 2009,
        2011, 2012, 2013, 2014,
        2016, 2017, 2018, 2019
    ]
    df = df.drop(columns=valores_no_deseados2, errors='ignore')
    return df

def calcular_porcentaje_pesocnttablafinal(df):
    # Calcula la suma del Trade Value para cada sección por año
    df['Total_por_Año'] = df.groupby('Year')['Volumen Total'].transform('sum')
    # Calcula el porcentaje de peso de cada sección en cada año
    df['Porcentaje_de_Peso'] = (df['Volumen Total'] / df['Total_por_Año'])* 100
    
    # Crea un nuevo DataFrame con el porcentaje de peso de cada sección en cada año
    nuevo_df = df[['Continent', 'Year', 'Porcentaje_de_Peso']].copy()
    pivoted_df = nuevo_df.pivot(index='Continent', columns='Year', values='Porcentaje_de_Peso').rename_axis(columns=None).reset_index()
    return pivoted_df

#######################################################################
######################################################################


#Grafíca
def Gráficaindustria(datos):
    # Agrupar los datos
    grafica = datos.groupby(by=['Section', 'Year']).agg({"Trade Value": 'sum'}).reset_index()

    # Elementos no deseados en 'Section'
    valores_no_deseados = ['Metals', 'Mineral Products', 'Miscellaneous', 'Paper Goods',
                           'Plastics and Rubbers', 'Precious Metals', 'Wood Products',
                           'Weapons', 'Instruments', 'Stone And Glass']

    # Columnas no deseadas
    valores_no_deseados2 = [
        1996, 1997, 1998, 1999,
        2000, 2001, 2002, 2003, 2004,
        2006, 2007, 2008, 2009,
        2011, 2012, 2013, 2014,
        2016, 2017, 2018, 2019
    ]

    # Filtrar los datos
    grafica = grafica[~grafica['Section'].isin(valores_no_deseados)]
    grafica = grafica[~grafica['Year'].isin(valores_no_deseados2)]

    # Plot con Seaborn
    g = sns.relplot(
        data=grafica,
        x="Year", y="Trade Value", col="Section", hue="Section",
        kind="line", palette="crest", linewidth=1.5, zorder=5,
        col_wrap=3, height=5, aspect=3, legend=True,
    )
    g.set(xlim=(1995, None))

    # Iterar sobre cada subgráfico para personalizarlo aún más
    for year, ax in g.axes_dict.items():
        # Añadir el año como anotación dentro del gráfico
        ax.text(.8, .85, year, transform=ax.transAxes, fontweight="bold")

        # Graficar cada serie temporal en el fondo
        sns.lineplot(
            data=grafica, x="Year", y="Trade Value", units="Section",
            estimator=None, color=".7", linewidth=2, ax=ax,
        )

    # Reducir la frecuencia de las marcas en el eje x
    for ax in g.axes.flat:
        ax.set_xticks(ax.get_xticks()[::2])

    # Ajustar los aspectos del gráfico
    g.set_titles("")
    g.set_axis_labels("", "Trade Value")
    g.tight_layout()

    # Mostrar el gráfico
    plt.show()
################################################################
#Grafíca
def Gráficaindustriatablafinal(datos):
    # Agrupar los datos
    grafica = datos.groupby(by=['Section', 'Year']).agg({"Volumen Total": 'sum'}).reset_index()

    # Elementos no deseados en 'Section'
    valores_no_deseados = ['Metals', 'Mineral Products', 'Miscellaneous', 'Paper Goods',
                           'Plastics and Rubbers', 'Precious Metals', 'Wood Products',
                           'Weapons', 'Instruments', 'Stone And Glass']

    # Columnas no deseadas
    valores_no_deseados2 = [
        1996, 1997, 1998, 1999,
        2000, 2001, 2002, 2003, 2004,
        2006, 2007, 2008, 2009,
        2011, 2012, 2013, 2014,
        2016, 2017, 2018, 2019
    ]

    # Filtrar los datos
    grafica = grafica[~grafica['Section'].isin(valores_no_deseados)]
    grafica = grafica[~grafica['Year'].isin(valores_no_deseados2)]

    # Plot con Seaborn
    g = sns.relplot(
        data=grafica,
        x="Year", y="Volumen Total", col="Section", hue="Section",
        kind="line", palette="crest", linewidth=1.5, zorder=5,
        col_wrap=3, height=5, aspect=3, legend=True,
    )
    g.set(xlim=(1995, None))

    # Iterar sobre cada subgráfico para personalizarlo aún más
    for year, ax in g.axes_dict.items():
        # Añadir el año como anotación dentro del gráfico
        ax.text(.8, .85, year, transform=ax.transAxes, fontweight="bold")

        # Graficar cada serie temporal en el fondo
        sns.lineplot(
            data=grafica, x="Year", y="Volumen Total", units="Section",
            estimator=None, color=".7", linewidth=2, ax=ax,
        )

    # Reducir la frecuencia de las marcas en el eje x
    for ax in g.axes.flat:
        ax.set_xticks(ax.get_xticks()[::2])

    # Ajustar los aspectos del gráfico
    g.set_titles("")
    g.set_axis_labels("", "Volumen Total")
    g.tight_layout()

    # Mostrar el gráfico
    plt.show()


#graficas simples  
# Exportaciones
    
def grafica_industriaexportaciones(df):
    years = df.columns[1:]
    colors = plt.cm.tab20.colors 
   
    plt.figure(figsize=(12, 8))

    for i, row in df.iterrows():
        Sector = row['Section']
        values = row[1:]  

        plt.plot(years, values, label=Sector, color=colors[i % len(colors)])

    mean_progression = df.iloc[:, 1:].mean(axis=0)
        

    plt.plot(years, mean_progression, label='Mean Progression', linestyle='--', linewidth=2, color='orange')

       
  
    plt.title('Evolución de las Exportaciones')
    plt.xlabel('Años')
    plt.ylabel('Valor (Dólares)')
    plt.legend()
    plt.xticks(rotation=45)  
    plt.tight_layout()
    return plt.show()

##################################################################################################################

# Importaciones
def grafica_industriaimportaciones(df):
    years = df.columns[1:]
    colors = plt.cm.tab10.colors 
   
    plt.figure(figsize=(12, 8))

    for i, row in df.iterrows():
        Sector = row['Section']
        values = row[1:]  

        
        plt.plot(years, values, label=Sector, color=colors[i % len(colors)])

    mean_progression = df.iloc[:, 1:].mean(axis=0)
        

    plt.plot(years, mean_progression, label='Mean Progression', linestyle='--', linewidth=2, color='orange')

       
  
    plt.title('Evolución de las importaciones')
    plt.xlabel('Años')
    plt.ylabel('Valor ( Billones de Dólares)')
    plt.legend()
    plt.xticks(rotation=45)  
    plt.tight_layout()
    return plt.show()


#Continentes exportaciones
def grafica_exportaciones(df):
    # Extraer años y colores
    years = df.columns[1:]
    colors = ['blue', 'green', 'red', 'purple', 'brown','orange','cyan']

    # Configurar el gráfico
    plt.figure(figsize=(12, 8))

    # Iterar sobre cada fila (correspondiente a un continente)
    for i, row in df.iterrows():
        Continent = row['Continent']
        values = row[1:]  # Obtener los valores de comercio para cada año

        # Graficar la serie temporal para cada continente
        plt.plot(years, values, label=Continent, color=colors[i % len(colors)])

    mean_progression = df.iloc[:, 1:].mean(axis=0)
        # Graficar la línea de la progresión media
    plt.plot(years, mean_progression, label='Mean Progression', linestyle='--', linewidth=2, color='orange')

       
    # Configuración adicional del gráfico
    plt.title('Evolution of Esports by Continent')
    plt.xlabel('Years')
    plt.ylabel('Trade Value')
    plt.legend()
    plt.xticks(rotation=45)  # Rotar los años para mejor visualización

    plt.tight_layout()
    return plt.show()

# Continentes importaciones
def grafica_importaciones(df):
    # Extraer años y colores
    years = df.columns[1:]
    colors = ['blue', 'green', 'red', 'purple', 'brown','orange','cyan']

    # Configurar el gráfico
    plt.figure(figsize=(12, 8))

    # Iterar sobre cada fila (correspondiente a un continente)
    for i, row in df.iterrows():
        Continent = row['Continent']
        values = row[1:]  # Obtener los valores de comercio para cada año

        # Graficar la serie temporal para cada continente
        plt.plot(years, values, label=Continent, color=colors[i % len(colors)])

    mean_progression = df.iloc[:, 1:].mean(axis=0)
        # Graficar la línea de la progresión media
    plt.plot(years, mean_progression, label='Mean Progression', linestyle='--', linewidth=2, color='orange')

       
    # Configuración adicional del gráfico
    plt.title('Evolution of Imports by Continent')
    plt.xlabel('Years')
    plt.ylabel('Trade Value')
    plt.legend()
    plt.xticks(rotation=45)  # Rotar los años para mejor visualización

    plt.tight_layout()
    return plt.show()

