# Exploratory data analysis (eda)_Análise explatória dos dados

import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Caminho do arquivo CSV e diretório de saída
file_path = r"C:\Users\natha\Downloads\ProjetoAM\CSV Agregado\dados_agregados.csv"
output_dir = r"C:\Users\natha\Downloads\ProjetoAM\CSV Agregado"

# Criar o diretório de saída, se não existir
os.makedirs(output_dir, exist_ok=True)

# Carregar o dataset
dados = pd.read_csv(file_path)

#remover linhas com valores ausentes
file_path.dropna(inplace=True)

# 1. Gráfico de Distribuição dos Casos por Ano (Gráfico de Linha)
plt.figure(figsize=(10, 6))

# Agrupando os dados para obter a contagem de casos por ano
casos_por_ano = dados.groupby('Ano').size().reset_index(name='Quantidade')

# Gráfico de linha
sns.lineplot(data=casos_por_ano, x='Ano', y='Quantidade', marker='o', palette='viridis')
plt.title("Distribuição dos Casos por Ano")
plt.xlabel("Ano")
plt.ylabel("Quantidade de Casos")
plt.xticks(rotation=45)
plt.tight_layout()

# Salvando o gráfico
plt.savefig(os.path.join(output_dir, "distribuicao_casos_por_ano.png"))
plt.close()

# 2. Relação entre Gênero e Agentes Etiológicos
plt.figure(figsize=(10, 6))
sns.countplot(data=dados, x='CS_SEXO', hue='CLASSI_FIN', palette='plasma')
plt.title("Relação entre Gênero e Agentes Etiológicos")
plt.xlabel("Gênero")
plt.ylabel("Quantidade de Casos")
plt.legend(title="Agentes Etiológicos")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "relacao_genero_agentes_etiologicos.png"))
plt.close()

# 3. Relação entre Raça e Agentes Etiológicos
plt.figure(figsize=(10, 6))
sns.countplot(data=dados, x='CS_RACA', hue='CLASSI_FIN', palette='cubehelix')
plt.title("Relação entre Raça e Agentes Etiológicos")
plt.xlabel("Raça")
plt.ylabel("Quantidade de Casos")
plt.legend(title="Agentes Etiológicos")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "relacao_raca_agentes_etiologicos.png"))
plt.close()

# 4. Comparação entre Fatores de Risco (Comorbidades)
comorbidades = ['CARDIOPATI', 'PNEUMOPATI', 'IMUNODEPRE', 'RENAL', 'OUT_MORBI']

# Contando as ocorrências das classes 1 e 2 para cada comorbidade
frequencias = dados[comorbidades].apply(lambda x: x.value_counts().get(1, 0) + x.value_counts().get(2, 0))

plt.figure(figsize=(12, 8))
frequencias.plot(kind='bar', color='teal')
plt.title("Comparação entre Fatores de Risco (Comorbidades) - Classes 1 e 2")
plt.xlabel("Comorbidades")
plt.ylabel("Quantidade de Ocorrências (Classes 1 e 2)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "comparacao_fatores_risco.png"))
plt.close()

# 5. Matriz de Estatísticas Descritivas
dados_filtrados = dados.drop(columns=['Ano'])  # Remove a coluna 'Ano'

# Calcula as estatísticas descritivas sem o 'count'
estatisticas_descritivas = dados_filtrados.describe().T
estatisticas_descritivas = estatisticas_descritivas.drop(columns=['count'])  # Remove a linha 'count'

plt.figure(figsize=(12, 8))
sns.heatmap(estatisticas_descritivas, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Matriz de Estatísticas Descritivas")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "matriz_estatisticas_descritivas.png"))
plt.close()

# 6. Curva de Densidade das Variáveis para Análise de Assimetria
plt.figure(figsize=(15, 10))

for i, column in enumerate(dados.columns, 1):
    plt.subplot(3, (len(dados.columns) + 2) // 3, i)  # Ajuste o layout
    sns.histplot(dados[column], bins=20, kde=False, color='skyblue', edgecolor='black', stat='density')
    
    # Calcula a curva normal
    mu, std = np.mean(dados[column]), np.std(dados[column])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2)  # Adiciona a curva normal em vermelho
    
    # Verifica a assimetria
    skewness = stats.skew(dados[column])
    if skewness > 0:
        plt.title(f"{column} (Assimetria Positiva)")
    elif skewness < 0:
        plt.title(f"{column} (Assimetria Negativa)")
    else:
        plt.title(f"{column} (Aproximadamente Gaussiana)")

plt.suptitle("Curva de Densidade das Variáveis", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_dir, "curva_densidade_variaveis.png"))
plt.close()
