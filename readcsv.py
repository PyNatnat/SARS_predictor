# Ler dados

import pandas as pd

# Caminho para o arquivo CSV
caminho_arquivo = r"C:\Users\natha\Downloads\ProjetoAM\CSV Agregado\dados_agregados.csv"

# Ler o arquivo CSV
dados = pd.read_csv(caminho_arquivo)

print(dados.head(100))

# Mostrar a quantidade de instâncias por valor único em cada coluna
for col in dados.columns:
    print(f"\nQuantidade de instâncias por valor na coluna {col}:")
    print(dados[col].value_counts())