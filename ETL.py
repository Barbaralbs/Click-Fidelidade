import pandas as pd

df = pd.read_csv('df_t.csv')

print("DataFrame carregado com sucesso!")
print(df.head())

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sqlalchemy import create_engine

# ----- ETAPA 1: EXTRAÇÃO (LOAD) -----
# Carregando a base de dados com o separador e o nome de arquivo corretos
try:
    df = pd.read_csv('df_t.csv', sep=';')
    print("DataFrame carregado com sucesso!")
except FileNotFoundError:
    print("Erro: O arquivo 'df_t.csv' não foi encontrado. Verifique o nome e a pasta.")

# ----- ETAPA 2: TRANSFORMAÇÃO (RFM) -----
# Convertendo a coluna gmv_success para tipo numérico
df['gmv_success'] = pd.to_numeric(df['gmv_success'], errors='coerce')

# Convertendo as colunas de data e hora para o tipo datetime
df['purchase_datetime'] = pd.to_datetime(df['date_purchase'] + ' ' + df['time_purchase'], format='%d/%m/%Y %H:%M:%S')

# Definindo uma data de referência para calcular a recência
current_date = pd.to_datetime('2025-09-10')

# Calculando RFM para cada cliente
rfm_df = df.groupby('fk_contact').agg(
    recency=('purchase_datetime', lambda x: (current_date - x.max()).days),
    frequency=('fk_contact', 'count'),
    monetary=('gmv_success', 'sum')
).reset_index()

# Renomeando a coluna de ID do cliente
rfm_df.rename(columns={'fk_contact': 'customer_id'}, inplace=True)

# ----- ETAPA 3: SEGMENTAÇÃO (K-MEANS) -----
# Removendo a coluna de ID antes do escalonamento
rfm_scaled = rfm_df.drop('customer_id', axis=1)

# Escalando os dados para o K-Means
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_scaled)

# Executando o K-Means com 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
kmeans.fit(rfm_scaled)

# Adicionando a coluna de segmento ao DataFrame
rfm_df['segment'] = kmeans.labels_

# Mapeando os números dos segmentos para nomes descritivos
rfm_df['segment'] = rfm_df['segment'].replace({
    0: 'Clientes em Risco',
    1: 'Clientes Fiéis',
    2: 'Clientes VIPs',
    3: 'Clientes Novos/Casuais'
})

# ----- ETAPA 4: CARREGAMENTO (LOAD) -----

# 1. Salvando o DataFrame em um arquivo CSV para o Power BI
rfm_df.to_csv('clientes_segmentados.csv', index=False, sep=';')
print("Análise concluída com sucesso! O arquivo 'clientes_segmentados.csv' foi criado.")

# 2. Salvando o DataFrame no SQL (SQLite)
try:
    db_file = 'clientes_segmentadooos.db'
    if os.path.exists(db_file):
        os.remove(db_file)
    engine = create_engine(f'sqlite:///{db_file}')
    rfm_df.to_sql('segmentacao_clientes', con=engine, if_exists='replace', index=False)
    print("\nDataFrame salvo com sucesso no SQL!")
except Exception as e:
    print(f"\nErro ao salvar no banco de dados: {e}")

# ----- IMPRIMINDO RESULTADOS FINAIS -----
print("\nPrimeiras 5 linhas do DataFrame RFM:")
print(rfm_df.head())