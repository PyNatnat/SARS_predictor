import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from imblearn.under_sampling import RandomUnderSampler

# Caminho do arquivo de dados
file_path = r"C:\Users\natha\Downloads\ProjetoAM\CSV Agregado\dados_agregados.csv"         

# Carregar os dados
df = pd.read_csv(file_path)

# Testar com valores ausentes imputados pela moda e pela exclusão de valores ausentes

#def preencher_com_moda(df):
    #df.replace('', pd.NA, inplace=True)
    #for column in df.columns:
        #moda = df[column].mode().iloc[0]
        #df[column] = df[column].fillna(moda)
    #return df

# Preencher os dados com a moda
#df = preencher_com_moda(df)

# remover linhas com valores ausentes ou NaN
df_filtered = df.dropna()

# Seleção das features e o target
features = [
    'CS_SEXO', 'NU_IDADE_N', 'CS_RACA', 'CS_ESCOL_N', 'CARDIOPATI', 
    'PNEUMOPATI', 'IMUNODEPRE', 'RENAL', 'OUT_MORBI'
]
target = 'CLASSI_FIN'

# Preparar os dados
X = df[features].values
y = df[target].values

# Aplicar undersampling
undersampler = RandomUnderSampler(random_state=27)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=27)

# Modelos
models = {
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=27),
    'Gaussian Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=27),
    'LinearSVC': LinearSVC(random_state=27),
    'Logistic Regression (OneVsRest)': OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000, random_state=27))
}

# Dicionário para armazenar as métricas
metrics = {
    'Accuracy': [],
    'Recall': [],
    'Precision': [],
    'F1-Score': []
}
model_names = []

# Avaliar e armazenar resultados dos modelos
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics['Accuracy'].append(accuracy_score(y_test, y_pred) * 100)
    metrics['Recall'].append(recall_score(y_test, y_pred, average='weighted') * 100)
    metrics['Precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100)
    metrics['F1-Score'].append(f1_score(y_test, y_pred, average='weighted') * 100)
    model_names.append(model_name)

# Converter o dicionário em DataFrame
metrics_df = pd.DataFrame(metrics, index=model_names)

# Gráficos
x = np.arange(len(metrics_df.columns)) * 2  # espaçamento entre grupos de métricas
width = 0.30  # largura das barras

fig, ax = plt.subplots(figsize=(12, 6))

# Plotar as barras com espaçamento entre os grupos de métricas
for i, model_name in enumerate(metrics_df.index):
    ax.bar(x + i * width, metrics_df.loc[model_name], width, label=model_name, edgecolor='black')

# Customizando labels e títulos
ax.set_xlabel('Métricas', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentual (%)', fontsize=14, fontweight='bold')
ax.set_title('Comparação das Métricas entre Modelos', fontsize=16, fontweight='bold')
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