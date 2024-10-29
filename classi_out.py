# análise do atributo classi_out (outros agentes etiológicos)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def processar_dados(file_path, output_folder):
    # Criar pasta de saída, se não existir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Ler o dataset
    df = pd.read_csv(file_path, engine='python', encoding='ISO-8859-1', delimiter=',')
    
    # Contagem dos valores únicos de CLASSI_OUT
    classi_out_counts = df['CLASSI_OUT'].value_counts(dropna=False)
    num_agentes_unicos = df['CLASSI_OUT'].nunique()
    
    # Filtragem para dados onde EVOLUCAO == 1.0 e colunas selecionadas
    df_filtered = df[df['EVOLUCAO'] == 1.0][['EVOLUCAO', 'CLASSI_FIN', 'CLASSI_OUT']]
    df_filtered.replace('', np.nan, inplace=True)
    
    # Mapeamento para CLASSI_OUT onde CLASSI_FIN == 3
    classi_out_3 = df_filtered[df_filtered['CLASSI_FIN'] == 3]
    agentes_unicos = classi_out_3['CLASSI_OUT'].dropna().unique()
    mapeamento_agentes = {agente: idx + 6 for idx, agente in enumerate(agentes_unicos)}
    df_filtered['CLASSI_OUT'] = df_filtered.apply(
        lambda row: mapeamento_agentes.get(row['CLASSI_OUT'], np.nan) if row['CLASSI_FIN'] == 3 else row['CLASSI_OUT'],
        axis=1
    )
    df_filtered.dropna(inplace=True)
    
    # Salvar resultado em uma planilha
    output_file_path = os.path.join(output_folder, 'Brasil_classi_out_resultado.xlsx')
    with pd.ExcelWriter(output_file_path) as writer:
        df_filtered.to_excel(writer, sheet_name='Dados Filtrados', index=False)
        classi_out_counts.to_excel(writer, sheet_name='Contagem CLASSI_OUT')
    
    # Plotar e salvar gráfico de contagem de CLASSI_OUT
    plt.figure(figsize=(10, 6))
    classi_out_counts.plot(kind='bar', color='skyblue')
    plt.title("Contagem de CLASSI_OUT")
    plt.xlabel("CLASSI_OUT")
    plt.ylabel("Contagem")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'contagem_classi_out.png'))
    
    print(f"Processamento concluído. Resultados salvos em: {output_folder}")
    print(f"Número de agentes etiológicos únicos em CLASSI_OUT: {num_agentes_unicos}")
    print("Mapa de agentes:", mapeamento_agentes)

# Caminho para o arquivo de entrada e a pasta de saída
file_path = r"C:\Users\natha\Downloads\ProjetoAM\CSV Brasil bruto\Brasil.csv"
output_folder = r"C:\Users\natha\Downloads\ProjetoAM\CLASSI_OUT"

# Executar o processamento
processar_dados(file_path, output_folder)