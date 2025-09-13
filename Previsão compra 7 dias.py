import pandas as pd


try:
    df = pd.read_csv('df_t.csv', sep=';')
except FileNotFoundError:
    print("Erro: O arquivo 'df_t.csv' não foi encontrado. Verifique o nome e a pasta.")

# Converte a coluna de data de compra para o formato datetime, se necessário.
df['date_purchase'] = pd.to_datetime(df['date_purchase'], format='%d/%m/%Y')

# Encontra a última data de compra
last_purchase_date = df['date_purchase'].max()

print("A última data de compra na sua base de dados é:")
print(last_purchase_date)

import pandas as pd
import numpy as np

# ----- ETAPA 1: CARREGAR DADOS -----
try:
    df = pd.read_csv('df_t.csv', sep=';')
    print("DataFrame carregado com sucesso!")
except FileNotFoundError:
    print("Erro: O arquivo 'df_t.csv' não foi encontrado. Verifique o nome e a pasta.")

# ----- ETAPA 2: PREPARAÇÃO DAS VARIÁVEIS -----
# Forçando a coluna 'gmv_success' a ser numérica
df['gmv_success'] = pd.to_numeric(df['gmv_success'], errors='coerce')

# Convertendo as colunas de data e hora para o formato datetime
df['date_purchase'] = pd.to_datetime(df['date_purchase'], format='%d/%m/%Y')
df['purchase_datetime'] = pd.to_datetime(df['date_purchase'].astype(str) + ' ' + df['time_purchase'])

# Definindo uma nova data de corte para garantir exemplos das duas classes
cutoff_date = pd.to_datetime('2024-01-01')
prediction_end_date = cutoff_date + pd.Timedelta(days=7)

# Separando os dados em treino e previsão
training_data = df[df['date_purchase'] <= cutoff_date]
prediction_data = df[(df['date_purchase'] > cutoff_date) & (df['date_purchase'] <= prediction_end_date)]

# ----- Criando as Variáveis de Entrada (X) -----
# Calculando o RFM para o período de treino
rfm_df = training_data.groupby('fk_contact').agg(
    recency=('purchase_datetime', lambda x: (cutoff_date - x.max()).days),
    frequency=('fk_contact', 'count'),
    monetary=('gmv_success', 'sum')
).reset_index()

# ----- Criando a Variável de Saída (y) -----
# Identificando clientes que compraram no período de previsão
customers_who_purchased = prediction_data['fk_contact'].unique()

# Criando a variável de saída (classificação binária)
rfm_df['purchased_in_next_7_days'] = rfm_df['fk_contact'].isin(customers_who_purchased).astype(int)

# ----- IMPRIMINDO RESULTADOS FINAIS -----
print("Tabela final pronta para o modelo. Primeiras 5 linhas:")
print(rfm_df.head())
print("\nContagem de clientes que compraram vs. que não compraram:")
print(rfm_df['purchased_in_next_7_days'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----- ETAPA 3: CONSTRUÇÃO DO MODELO -----

# Definindo as variáveis de entrada (X) e a variável de saída (y)
X = rfm_df[['recency', 'frequency', 'monetary']]
y = rfm_df['purchased_in_next_7_days']

# Dividindo os dados em conjuntos de treino e teste
# Usaremos 80% dos dados para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Criando e treinando o modelo de Regressão Logística
# Adicionando o parâmetro class_weight='balanced' para balancear as classes
model = LogisticRegression(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

print("Modelo treinado e previsões realizadas com sucesso!")

# ----- ETAPA 4: AVALIAÇÃO DO MODELO -----

# Calculando a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do modelo: {accuracy:.4f}")

# Exibindo a matriz de confusão para uma análise mais detalhada
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(cm)

# Exibindo o relatório de classificação para precisão e recall
cr = classification_report(y_test, y_pred, zero_division=0)
print("\nRelatório de Classificação:")
print(cr)

import pandas as pd
from datetime import datetime

# --- Parte 1: Salvando as Previsões em uma Planilha CSV ---

# Criando um DataFrame com os resultados da previsão
results_df = X_test.copy()
results_df['predicted_purchase'] = y_pred
results_df['actual_purchase'] = y_test
results_df['customer_id'] = X_test.index # Adicionando o customer_id ao DataFrame

# Renomeando as colunas para maior clareza
results_df.rename(columns={'predicted_purchase': 'previsao_compra', 'actual_purchase': 'compra_real'}, inplace=True)

# Salvando o DataFrame como um arquivo CSV com data e hora atual
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
predictions_filename = f"previsoes_7_dias_{timestamp}.csv"
results_df.to_csv(predictions_filename, index=False, sep=';')

print("\n")
print(f"Planilha de previsões criada com sucesso: {predictions_filename}")
print("Esta planilha pode ser carregada em um banco de dados.")

# --- Parte 2: Salvando as Métricas do Modelo em um Documento ---

# Criando o conteúdo do arquivo de métricas
metrics_content = f"""
Relatório de Desempenho do Modelo - Previsão de Compra em 7 Dias

Data de Execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Métricas do Modelo:
-------------------
Acurácia: {accuracy:.4f}

Matriz de Confusão:
(verdadeiro negativo, falso positivo)
(falso negativo, verdadeiro positivo)
{cm}

Relatório de Classificação:
{cr}
"""

# Salvando o conteúdo em um arquivo de texto
metrics_filename = f"metrica_modelo_{timestamp}.txt"
with open(metrics_filename, "w") as f:
    f.write(metrics_content)

print(f"Documento de métricas criado com sucesso: {metrics_filename}")