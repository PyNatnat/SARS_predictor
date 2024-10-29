''' Arquivo para fazer o tratamento dos csvs'''

import os
import pandas as pd
import numpy as np

def tratarIdade(idade):
    if pd.notna(idade):
        try:
            idade_str = str(int(idade))
            if len(idade_str) == 4:  
                idade = int(idade_str[-2:])
            if idade <= 14:
                return 0  # jovem
            elif 15 <= idade <= 64:
                return 1  # adulto
            elif idade >= 65:
                return 2  # idoso
        except ValueError:
            pass
    return idade

def tratarSexo(sexo):
    if sexo == 'M':  # masculino
        return 1
    elif sexo == 'F':  # feminino
        return 2
    elif sexo == 'I':  # ignorado
        return 9
    return sexo

def tratarEscola(escola, ano):
    if ano in ['2009', '2010', '2011', '2012','2013', '2014', '2015', '2016', '2017', '2018']:
        if escola == '2.0':
            return 3
        elif escola == '3.0':
            return 4
    return escola

def filtroFeatures(input_csv, output_csv):
    # Features que queremos usar
    features = [
        'CS_SEXO', 'NU_IDADE_N', 'CS_RACA', 'CS_ESCOL_N', 'PUERPERA',
        'CARDIOPATI', 'SIND_DOWN', 'HEPATICA', 'NEUROLOGIC', 'PNEUMOPATI', 
        'IMUNODEPRE', 'RENAL', 'OBESIDADE', 'OUT_MORBI', 'EVOLUCAO', 'CLASSI_FIN'
    ]
    
    # Ler o arquivo CSV
    df = pd.read_csv(input_csv, engine='python', encoding='ISO-8859-1', delimiter=';')

    # Determinar o ano a partir do nome do arquivo
    ano_str = os.path.basename(input_csv)[-6:-4]
    ano = 2000 + int(ano_str)

    # Adicionar a coluna 'Ano'
    df['Ano'] = ano

    # Filtrar as colunas desejadas e criar uma cópia
    df_filtered = df[features + ['Ano']].copy()

    # Filtrar por não evoluiu a óbito
    df_filtered = df_filtered[df_filtered['EVOLUCAO'] == 1.0]

    # Substituir strings vazias por NaN 
    df_filtered.replace('', np.nan, inplace=True)

    # Aplicar o tratamento nas colunas
    df_filtered['NU_IDADE_N'] = df_filtered['NU_IDADE_N'].apply(tratarIdade)
    df_filtered['CS_SEXO'] = df_filtered['CS_SEXO'].apply(tratarSexo)
    df_filtered['CS_ESCOL_N'] = df_filtered['CS_ESCOL_N'].apply(lambda escola: tratarEscola(escola, ano))

    # Salvar o novo DataFrame filtrado em um novo arquivo CSV
    df_filtered.to_csv(output_csv, index=False)
    
    return df_filtered

def processarArquivosNaPasta(input_folder, output_folder):
    # Verifica se a pasta de saída existe, se não tiver, cria
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Processa cada arquivo CSV
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_csv = os.path.join(input_folder, filename)
            output_csv = os.path.join(output_folder, filename.replace('.csv', '_filtrado.csv'))
            try:
                filtroFeatures(input_csv, output_csv)
                print(f"Processado: {filename}")
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")

def agregarArquivos(output_folder, output_subfolder):
    full_output_folder = os.path.join(output_folder, output_subfolder)
    if not os.path.exists(full_output_folder):
        os.makedirs(full_output_folder)

    dfs = []

    # Verifica todos os arquivos CSV na pasta de saída
    for filename in os.listdir(output_folder):
        if filename.endswith("_filtrado.csv"):
            file_path = os.path.join(output_folder, filename)
            try:
                df = pd.read_csv(file_path, engine='python', encoding='ISO-8859-1', delimiter=',')
                df.dropna(axis=1, how='all', inplace=True)
                df.dropna(inplace=True)
                dfs.append(df)
                print(f"Adicionado: {filename}")
            except Exception as e:
                print(f"Erro ao agregar {filename}: {e}")

    # Junta todos os DataFrames em um único DataFrame
    df_aggregated = pd.concat(dfs, ignore_index=True)

    # Define o caminho para o arquivo agregado
    output_file = os.path.join(full_output_folder, 'dados_agregados.csv')
    df_aggregated.to_csv(output_file, index=False)

    return df_aggregated

# Caminhos de entrada e saída
input_folder = r"C:\Users\natha\Downloads\ProjetoAM\SARS_Brazil"
output_folder = r"C:\Users\natha\Downloads\ProjetoAM"
processarArquivosNaPasta(input_folder, output_folder)

# Agrupar todos os CSVs em um só
output_subfolder = 'CSV Agregado'
agregarArquivos(output_folder, output_subfolder)