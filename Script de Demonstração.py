import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
import os

warnings.filterwarnings('ignore')

print("--- DEMONSTRAÇÃO DA ARQUITETURA DE SOLUÇÃO ---")

# ----- ETAPA 1: EXECUÇÃO ETL (EXTRAÇÃO E LIMPEZA) -----
print("\n[ETAPA 1/3] Executando ETL: Coleta de Dados e Limpeza...")
try:
    df = pd.read_csv('df_t.csv', sep=';')
except FileNotFoundError:
    print("Erro: O arquivo 'df_t.csv' não foi encontrado.")
    exit()

df.dropna(subset=['fk_contact', 'date_purchase', 'gmv_success'], inplace=True)
df['gmv_success'] = pd.to_numeric(df['gmv_success'], errors='coerce')
df.dropna(subset=['gmv_success'], inplace=True)

df['date_purchase'] = pd.to_datetime(df['date_purchase'], format='%d/%m/%Y')
df['purchase_datetime'] = pd.to_datetime(df['date_purchase'].astype(str) + ' ' + df['time_purchase'])
df['route'] = df['place_origin_departure'] + ' - ' + df['place_destination_departure']
print("Dados carregados e limpos com sucesso!")

# ----- ETAPA 2: MODELAGEM E ARMAZENAMENTO FINAL (SEGMENTAÇÃO) -----
print("\n[ETAPA 2/3] Modelagem e Armazenamento Final: Segmentação...")

# Calculando as métricas RFM de forma segura
rfm_df = df.groupby('fk_contact').agg(
    recency=('purchase_datetime', lambda x: (df['purchase_datetime'].max() - x.max()).days),
    frequency=('fk_contact', 'count'),
    monetary=('gmv_success', 'sum')
).reset_index()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
rfm_scaled = rfm_df.drop('fk_contact', axis=1)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_scaled)
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
kmeans.fit(rfm_scaled)
rfm_df['segment'] = kmeans.labels_
rfm_df['segment'] = rfm_df['segment'].replace({0: 'Clientes em Risco', 1: 'Clientes Fiéis', 2: 'Clientes VIPs', 3: 'Clientes Novos/Casuais'})

db_file = 'dados_analiticos.db'
if os.path.exists(db_file): os.remove(db_file)
engine = create_engine(f'sqlite:///{db_file}')
rfm_df.to_sql('segmentacao_clientes', con=engine, if_exists='replace', index=False)
print("Dados de segmentação carregados com sucesso no SQL!")

# ----- ETAPA 3: COMUNICAÇÃO (PREVISÃO) -----
print("\n[ETAPA 3/3] Análise Preditiva para Comunicação...")
popular_routes = df['route'].value_counts().nlargest(50).index
df_filtered = df[df['route'].isin(popular_routes)]
df_sorted = df_filtered.sort_values(['fk_contact', 'date_purchase'])
df_sorted['next_route'] = df_sorted.groupby('fk_contact')['route'].shift(-1)
df_sorted['days_until_next_purchase'] = (df_sorted.groupby('fk_contact')['date_purchase'].shift(-1) - df_sorted['date_purchase']).dt.days
combined_df = df_sorted.dropna(subset=['next_route', 'days_until_next_purchase']).drop_duplicates(subset=['fk_contact'], keep='last')

X_class = combined_df[['route']]
y_class = combined_df['next_route']
encoder_y_class = LabelEncoder()
y_encoded_class = encoder_y_class.fit_transform(y_class)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_encoded_class, test_size=0.2, random_state=42)
encoder_x_class = LabelEncoder()
X_train_class_encoded = encoder_x_class.fit_transform(X_train_class.squeeze()).reshape(-1, 1)
X_test_class_encoded = encoder_x_class.transform(X_test_class.squeeze()).reshape(-1, 1)
model_class = LogisticRegression(random_state=42, n_jobs=-1, solver='lbfgs', max_iter=1000)
model_class.fit(X_train_class_encoded, y_train_class)
y_pred_class = model_class.predict(X_test_class_encoded)
y_pred_routes = encoder_y_class.inverse_transform(y_pred_class)
predictions_df = pd.DataFrame({'fk_contact': X_test_class.squeeze(), 'predicted_route': y_pred_routes})

predictions_df.to_sql('previsoes_marketing', con=engine, if_exists='replace', index=False)
print("Dados de previsão carregados com sucesso no SQL!")

print("\n--- DEMONSTRAÇÃO CONCLUÍDA! ---")
print("Você pode agora conectar o Power BI ao arquivo 'dados_analiticos.db' para a visualização, e a tabela 'previsoes_marketing' pode ser usada por um script de comunicação para enviar e-mails.")