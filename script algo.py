import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ----- ETAPA 1: CARREGAR E PREPARAR DADOS -----
try:
    df = pd.read_csv('df_t.csv', sep=';')
except FileNotFoundError:
    print("Erro: O arquivo 'df_t.csv' não foi encontrado. Verifique o nome e a pasta.")
df.dropna(subset=['fk_contact', 'date_purchase', 'gmv_success'], inplace=True)
df['gmv_success'] = pd.to_numeric(df['gmv_success'], errors='coerce')
df.dropna(subset=['gmv_success'], inplace=True)
df['date_purchase'] = pd.to_datetime(df['date_purchase'], format='%d/%m/%Y')
df['purchase_datetime'] = pd.to_datetime(df['date_purchase'].astype(str) + ' ' + df['time_purchase'])
df['route'] = df['place_origin_departure'] + ' - ' + df['place_destination_departure']

# ----- ETAPA 2: CRIAR VARIÁVEIS PARA O MODELO -----
popular_routes = df['route'].value_counts().nlargest(50).index
df_filtered = df[df['route'].isin(popular_routes)]
df_sorted = df_filtered.sort_values(['fk_contact', 'date_purchase'])
df_sorted['next_route'] = df_sorted.groupby('fk_contact')['route'].shift(-1)
df_sorted['days_until_next_purchase'] = (df_sorted.groupby('fk_contact')['date_purchase'].shift(-1) - df_sorted['date_purchase']).dt.days
combined_df = df_sorted.dropna(subset=['next_route', 'days_until_next_purchase']).drop_duplicates(subset=['fk_contact'], keep='last')
combined_df = combined_df[['fk_contact', 'route', 'next_route', 'days_until_next_purchase']]

# ----- ETAPA 3: CONSTRUÇÃO E AVALIAÇÃO DO MODELO -----
X = combined_df[['route']]
y = combined_df['next_route']
encoder_y = LabelEncoder()
y_encoded = encoder_y.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
encoder_x = LabelEncoder()
X_train_encoded = encoder_x.fit_transform(X_train.squeeze()).reshape(-1, 1)
X_test_encoded = encoder_x.transform(X_test.squeeze()).reshape(-1, 1)
model = LogisticRegression(random_state=42, n_jobs=-1, solver='lbfgs', max_iter=1000)
model.fit(X_train_encoded, y_train)
y_pred_encoded = model.predict(X_test_encoded)
y_pred_routes = encoder_y.inverse_transform(y_pred_encoded)
y_test_routes = encoder_y.inverse_transform(y_test)

# ----- ETAPA 4: SALVANDO O RESULTADO FINAL -----
results_df = pd.DataFrame()
results_df['fk_contact'] = X_test['route'].reset_index(drop=True)
results_df['predicted_route'] = y_pred_routes
results_df['actual_route'] = y_test_routes
results_df.to_csv('previsoes_trecho_final.csv', index=False, sep=';')

print("Planilha de previsões final criada com sucesso!")