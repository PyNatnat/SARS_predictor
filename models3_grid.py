# gridsearch - busca pelos melhores hiperparâmetros

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler

# Caminho do arquivo de dados
file_path = r"C:\Users\natha\Downloads\ProjetoAM\CSV Agregado\dados_agregados.csv"

# Carregar os dados
df = pd.read_csv(file_path)

def preencher_com_moda(df):
    df.replace('', pd.NA, inplace=True)
    for column in df.columns:
        moda = df[column].mode().iloc[0]
        df[column] = df[column].fillna(moda)
    return df

# Preencher os dados com a moda
df = preencher_com_moda(df)

# remover linhas com valores ausentes ou NaN
#df_filtered = df.dropna()

# Seleção das features e o target
features = [
    'CS_SEXO', 'NU_IDADE_N', 'CS_RACA', 'CS_ESCOL_N', 'CARDIOPATI', 
    'PNEUMOPATI', 'IMUNODEPRE', 'RENAL', 'OUT_MORBI'
]
target = 'CLASSI_FIN'

# Preparar os dados
X = df[features].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

# Escalonar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelos com hiperparâmetros padrão
models = {
    'LinearSVC': LinearSVC(),
    'Logistic Regression (OneVsRest)': OneVsRestClassifier(LogisticRegression(max_iter=200)),  # Aumentando max_iter
    'KNN': KNeighborsClassifier()
}

# Dicionário para armazenar as métricas
metrics = {
    'Accuracy': [],
    'Recall': [],
    'Precision': [],
    'F1-Score': []
}
model_names = []

# Hiperparâmetros para Grid Search
param_grid = {
    'LinearSVC': {
        'C': [0.01, 0.1, 1, 10, 100]
    },
    'Logistic Regression (OneVsRest)': {
        'estimator__C': [0.01, 0.1, 1, 10, 100]
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11]
    }
}

# Avaliar e armazenar resultados dos modelos com Grid Search
for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics['Accuracy'].append(accuracy_score(y_test, y_pred) * 100)
    metrics['Recall'].append(recall_score(y_test, y_pred, average='weighted') * 100)
    metrics['Precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100)
    metrics['F1-Score'].append(f1_score(y_test, y_pred, average='weighted') * 100)
    model_names.append(model_name)

# Converter o dicionário em DataFrame
metrics_df = pd.DataFrame(metrics, index=model_names)

# Gráficos
x = np.arange(len(metrics_df.columns)) * 2  # espaçamento entre grupos de métricas
width = 0.30  # aumentar a largura das barras

fig, ax = plt.subplots(figsize=(12, 6))

# Plotar as barras com espaçamento entre os grupos de métricas
for i, model_name in enumerate(metrics_df.index):
    ax.bar(x + i * width, metrics_df.loc[model_name], width, label=model_name, edgecolor='black')

# Customizando labels e títulos
ax.set_xlabel('Métricas', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentual (%)', fontsize=14, fontweight='bold')
ax.set_title('Comparação das Métricas entre Modelos (Grid Search)', fontsize=16, fontweight='bold')
ax.set_xticks(x + width * (len(model_names) - 1) / 2)
ax.set_xticklabels(metrics_df.columns)
ax.tick_params(axis='x', labelsize=12) 
ax.tick_params(axis='y', labelsize=12)  

# Adicionar a legenda
ax.legend(title='Modelos', title_fontsize='13', fontsize='12')

# Ajuste do limite do eixo y para caber as anotações
ax.set_ylim(0, 120)

# Mostrar os valores acima das barras de forma vertical
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10,  # Tamanho do texto
                    fontweight='bold',  # Negrito
                    rotation=90)  # Texto na vertical

# Adicionar valores para cada barra
for model_name in model_names:
    autolabel(ax.patches[model_names.index(model_name) * len(metrics_df.columns): (model_names.index(model_name) + 1) * len(metrics_df.columns)])

fig.tight_layout()
plt.show()