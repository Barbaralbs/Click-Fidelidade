import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ----- ETAPA 1: CARREGAR E PREPARAR DADOS -----
try:
    df = pd.read_csv('df_t.csv', sep=';')
except FileNotFoundError:
    print("Erro: O arquivo 'df_t.csv' não foi encontrado. Verifique o nome e a pasta.")

df.dropna(subset=['fk_contact', 'date_purchase', 'place_origin_departure', 'place_destination_departure', 'gmv_success'], inplace=True)
df['gmv_success'] = pd.to_numeric(df['gmv_success'], errors='coerce')
df.dropna(subset=['gmv_success'], inplace=True)
df['date_purchase'] = pd.to_datetime(df['date_purchase'], format='%d/%m/%Y')
df['purchase_datetime'] = pd.to_datetime(df['date_purchase'].astype(str) + ' ' + df['time_purchase'])
df['route'] = df['place_origin_departure'] + ' - ' + df['place_destination_departure']

# ----- ETAPA 2: PREPARAR OS DADOS PARA AMBOS OS MODELOS -----
cutoff_date = pd.to_datetime('2024-04-01')

df_filtered = df[df['route'].isin(df['route'].value_counts().nlargest(50).index)]
df_sorted = df_filtered.sort_values(['fk_contact', 'date_purchase'])
df_sorted['next_route'] = df_sorted.groupby('fk_contact')['route'].shift(-1)
df_sorted['days_until_next_purchase'] = (df_sorted.groupby('fk_contact')['date_purchase'].shift(-1) - df_sorted['date_purchase']).dt.days

combined_df = df_sorted.dropna(subset=['next_route', 'days_until_next_purchase']).drop_duplicates(subset=['fk_contact'], keep='last')
combined_df = combined_df[['fk_contact', 'route', 'next_route', 'days_until_next_purchase', 'gmv_success']]

# ----- ETAPA 3: CONSTRUÇÃO DOS MODELOS -----

# Modelo de Classificação (Prever o Trecho)
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

# Modelo de Regressão (Prever a Data)
X_reg = combined_df[['route']]
y_reg = combined_df['days_until_next_purchase']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
encoder_x_reg = LabelEncoder()
X_train_reg_encoded = encoder_x_reg.fit_transform(X_train_reg.squeeze()).reshape(-1, 1)
X_test_reg_encoded = encoder_x_reg.transform(X_test_reg.squeeze()).reshape(-1, 1)
model_reg = KNeighborsRegressor(n_neighbors=5)
model_reg.fit(X_train_reg_encoded, y_train_reg)
y_pred_reg = model_reg.predict(X_test_reg_encoded)

# ----- ETAPA 4: COMBINAR OS RESULTADOS E DOCUMENTAR -----
results_df = pd.DataFrame(X_test_class, columns=['last_route_name'])
results_df['predicted_route'] = encoder_y_class.inverse_transform(y_pred_class)
results_df['actual_route'] = encoder_y_class.inverse_transform(y_test_class)
results_df['predicted_days'] = np.round(y_pred_reg, 0)
results_df['actual_days'] = y_test_reg.values
results_df['fk_contact'] = X_test.index.values

print("\nRelatório de Classificação (Previsão de Trecho):")
print(classification_report(y_test_class, y_pred_class, zero_division=0))

print("\nRelatório de Regressão (Previsão de Dias):")
print(f"Erro Médio Absoluto (MAE): {mean_absolute_error(y_test_reg, y_pred_reg):.2f} dias")

print("\nPrimeiras 5 previsões combinadas:")
print(results_df.head())

results_df.to_csv('previsoes_finais.csv', index=False, sep=';')
print("\nPrevisões finais salvas em 'previsoes_finais.csv'.")