"""
ETAPA 01 - Treinamento do Modelo de Previsão de Atraso
Classificação binária com XGBoost

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Caminhos portáteis
BASE_DIR  = Path(__file__).resolve().parent.parent  # raiz do projeto
DATA_DIR  = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Pasta 'data/' não encontrada em: {BASE_DIR}\n"
        "Execute feature_engineering.py antes deste script."
    )

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("  ETAPA 01 - TREINAMENTO: PREVISÃO DE ATRASO NA ENTREGA")
print("=" * 80)

# 1. Carregar dataset
print("\n[1/6] Carregando dataset...")

dataset_path = DATA_DIR / 'etapa01_delay_dataset.csv'
if not dataset_path.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {dataset_path}\n"
        "Execute feature_engineering.py antes deste script."
    )

df = pd.read_csv(dataset_path, parse_dates=['order_purchase_timestamp'])

if 'is_delayed' not in df.columns:
    raise ValueError(
        "Coluna 'is_delayed' não encontrada no dataset.\n"
        "Verifique se o feature_engineering.py foi executado corretamente."
    )

print(f"  ✓ Dataset carregado: {df.shape}")
n_total   = len(df)
n_delayed = df['is_delayed'].sum()
print(f"  Target distribution:")
print(f"    Atrasados:  {n_delayed:,}  ({n_delayed / n_total * 100:.1f}%)")
print(f"    No prazo:   {n_total - n_delayed:,}  ({(n_total - n_delayed) / n_total * 100:.1f}%)")

# 2. Selecionar e preparar features
print("\n[2/6] Selecionando e preparando features...")

FEATURE_COLS = [
    # Temporais (apenas da compra — sem data leakage)
    'purchase_year', 'purchase_month', 'purchase_day',
    'purchase_weekday', 'purchase_hour', 'purchase_quarter',

    # Prazo prometido
    'promised_delivery_days',

    # Características do pedido
    'order_item_id',          # número de itens
    'price', 'freight_value', 'payment_value_total',
    'payment_installments_max',

    # Características do produto
    'product_weight_g', 'product_volume_cm3',
    'product_photos_qty',

    # Geografia e distância
    'distance_customer_seller_km', 'same_state',

    # Categóricas
    'customer_state', 'seller_state',
    'product_category_name_english', 'payment_type_main'
]

# Filtrar apenas features disponíveis no dataset
feature_cols = [col for col in FEATURE_COLS if col in df.columns]
missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
print(f"  Features selecionadas: {len(feature_cols)}")
if missing_cols:
    print(f"  ⚠ Features ausentes (ignoradas): {missing_cols}")


# 3. Split temporal (time-based)
print("\n[3/6] Dividindo dataset (time-based split)...")

# Ordenar por data de compra para garantir split temporal correto
if 'order_purchase_timestamp' in df.columns:
    df = df.sort_values('order_purchase_timestamp').reset_index(drop=True)
    print("  ✓ Dataset ordenado por order_purchase_timestamp")
else:
    print("  ⚠ Coluna 'order_purchase_timestamp' não encontrada — split sem ordenação temporal.")

X = df[feature_cols].copy()
y = df['is_delayed'].values

# Tratar valores nulos
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
for col in cat_cols:
    X[col] = X[col].fillna('unknown')

# Encoding de variáveis categóricas com mapeamento robusto (evita unseen labels)
label_encoders = {}
for col in cat_cols:
    categories = X[col].unique()
    mapping = {cat: idx for idx, cat in enumerate(sorted(categories))}
    X[col] = X[col].map(mapping).fillna(-1).astype(int)
    label_encoders[col] = mapping  # salvar mapeamento para inferência

# Split: últimos 20% como teste
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"  Treino: {X_train.shape}  |  Teste: {X_test.shape}")
print(f"  Proporção atraso — treino: {y_train.mean() * 100:.1f}%  |  teste: {y_test.mean() * 100:.1f}%")

# 4. Treinar modelo XGBoost
print("\n[4/6] Treinando modelo XGBoost...")

# scale_pos_weight para balancear classes (com proteção contra divisão por zero)
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
print(f"  scale_pos_weight: {scale_pos_weight:.2f}  (neg={n_neg:,} / pos={n_pos:,})")

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print("  ✓ Modelo treinado com sucesso!")

# 5. Avaliar modelo
print("\n[5/6] Avaliando modelo...")

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
auc_roc   = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 80)
print("  MÉTRICAS DE AVALIAÇÃO — ETAPA 01")
print("=" * 80)
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc_roc:.4f}")

print("\n  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Delay', 'Delayed']))

print("\n  Top 10 Features Mais Importantes:")
feature_importance = (
    pd.DataFrame({
        'feature':    feature_cols,
        'importance': model.feature_importances_
    })
    .sort_values('importance', ascending=False)
    .head(10)
    .reset_index(drop=True)
)
feature_importance['importance'] = feature_importance['importance'].round(4)
print(feature_importance.to_string(index=False))

# 6. Salvar modelo e artefatos
print("\n[6/6] Salvando modelo e artefatos...")

joblib.dump(model,          MODEL_DIR / 'etapa01_delay_model.pkl')
joblib.dump(label_encoders, MODEL_DIR / 'etapa01_label_encoders.pkl')
joblib.dump(feature_cols,   MODEL_DIR / 'etapa01_feature_cols.pkl')

# Salvar métricas com timestamp (não sobrescreve runs anteriores)
metrics = {
    'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'train_size':  len(X_train),
    'test_size':   len(X_test),
    'accuracy':    round(accuracy,  4),
    'precision':   round(precision, 4),
    'recall':      round(recall,    4),
    'f1_score':    round(f1,        4),
    'auc_roc':     round(auc_roc,   4),
}
pd.DataFrame([metrics]).to_csv(MODEL_DIR / 'etapa01_metrics.csv', index=False)

print(f"  ✓ etapa01_delay_model.pkl")
print(f"  ✓ etapa01_label_encoders.pkl")
print(f"  ✓ etapa01_feature_cols.pkl")
print(f"  ✓ etapa01_metrics.csv")
print(f"  Diretório: {MODEL_DIR}")

print("\n" + "=" * 80)
print("  ETAPA 01 CONCLUÍDA COM SUCESSO!")
print("=" * 80)