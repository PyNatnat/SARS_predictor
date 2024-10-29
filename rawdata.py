# Dados brutos com apenas os atributos selecionados (mantendo valores ausentes)

# inclusão de SG_UF_NOT para contar as instâncias de estados/regiões do Brasil
# inclusão de CLASSI_OUT (para investigação)

import os
import pandas as pd
import numpy as np

def filtroFeatures(input_csv, output_csv):
    # Atributos que serão selecionados 
    features = [
        'SG_UF_NOT','CS_SEXO', 'NU_IDADE_N', 'CS_RACA', 'CS_ESCOL_N', 'PUERPERA',
        'CARDIOPATI', 'SIND_DOWN', 'HEPATICA', 'NEUROLOGIC', 'PNEUMOPATI',
        'IMUNODEPRE', 'RENAL', 'OBESIDADE', 'OUT_MORBI', 'EVOLUCAO', 'CLASSI_FIN', 'CLASSI_OUT'
    ]
    
    # Lendo o arquivo CSV
    df = pd.read_csv(input_csv, engine='python', encoding='ISO-8859-1', delimiter=';')

    # Filtrar por não evoluiu a óbito
    df_filtered = df[df['EVOLUCAO'] == 1.0]

    # Filtrar as colunas desejadas
    df_filtered = df_filtered[features]

    # Substituir strings vazias por NaN
    df_filtered.replace('', np.nan, inplace=True)

    # Remover linhas com valores ausentes ou NaN
    #df_filtered = df_filtered.dropna()

    # Salvar o DataFrame filtrado em um arquivo CSV
    df_filtered.to_csv(output_csv, mode='w', index=False)  # Mudança para 'w' para sobrescrever o arquivo

    print(f"Arquivo filtrado salvo em {output_csv}")

def processarArquivosNaPasta(input_folder, output_folder):
    # Criar a pasta de saída se não existir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterar pelos arquivos na pasta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_csv = os.path.join(input_folder, filename)
            output_csv = os.path.join(output_folder, filename.replace('.csv', '_filtrado.csv'))
            try:
                # Aplicar a função de filtro em cada arquivo
                filtroFeatures(input_csv, output_csv)
                print(f"Processado: {filename}")
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")

def agregarArquivos(output_folder, output_subfolder):
    # Criar a subpasta de saída se não existir
    full_output_folder = os.path.join(output_folder, output_subfolder)
    if not os.path.exists(full_output_folder):
        os.makedirs(full_output_folder)

    dfs = []  # Lista para armazenar DataFrames
    for filename in os.listdir(output_folder):
        if filename.endswith("_filtrado.csv"):
            file_path = os.path.join(output_folder, filename)
            try:
                # Ler o DataFrame do arquivo filtrado
                df = pd.read_csv(file_path, engine='python', encoding='ISO-8859-1', delimiter=',')
                dfs.append(df)  # Adicionar o DataFrame à lista
                print(f"Adicionado: {filename}")
            except Exception as e:
                print(f"Erro ao agregar {filename}: {e}")

    # Agregar todos os DataFrames em um único DataFrame
    if dfs:
        df_aggregated = pd.concat(dfs, ignore_index=True, sort=False)
        output_file = os.path.join(full_output_folder, 'Brasil.csv')  # Nome do arquivo de saída
        df_aggregated.to_csv(output_file, index=False)  # Salvar o DataFrame agregado em um arquivo CSV
        print(f"Arquivo agregado salvo em {output_file}")
    else:
        print("Nenhum arquivo para agregar.")

# pasta de entrada com csv brutos
input_folder = r"C:\Users\natha\Downloads\ProjetoAM\SARS_Brazil"

# pasta de saída com cvs tratados
output_folder = r"C:\Users\natha\Downloads\ProjetoAM"

processarArquivosNaPasta(input_folder, output_folder)

# Agrupar todos os CSV em um só
output_subfolder = 'CSV Brasil bruto'
agregarArquivos(output_folder, output_subfolder)