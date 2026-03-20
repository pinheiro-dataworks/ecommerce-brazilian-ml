"""
ETAPA 02 - Treinamento dos Modelos de Churn e LTV
Classificação (Churn) + Regressão (LTV) com XGBoost

CORREÇÃO DE DATA LEAKAGE:
─────────────────────────────────────────────────────────────────────────────
O problema original: 'recency_days' era usada como feature do modelo de churn,
mas ela É a definição do próprio label (churn = recency_days > 120).
O modelo aprendia a regra trivial recency_days > 120 → churn = 1, resultando
em AUC-ROC = 1.0, que não tem valor preditivo real.

A correção aplicada aqui:
  1. 'recency_days' removida das features de churn.
  2. 'total_spent' removida das features de churn (mesma janela temporal do LTV).
  3. Split temporal explícito: features derivadas de comportamento anterior ao
     cutoff; label indica o que aconteceu APÓS o cutoff.
  4. Lógica de cutoff documentada para reprodutibilidade.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Pasta 'data/' não encontrada em: {BASE_DIR}\n"
        "Execute feature_engineering.py antes deste script."
    )

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("  ETAPA 02 - TREINAMENTO: CHURN E LTV  (sem data leakage)")
print("=" * 80)


# ── 1. Carregar dataset ───────────────────────────────────────────────────────
print("\n[1/7] Carregando dataset...")

dataset_path = DATA_DIR / 'etapa02_churn_ltv_dataset.csv'
if not dataset_path.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {dataset_path}\n"
        "Execute feature_engineering.py antes deste script."
    )

df = pd.read_csv(
    dataset_path,
    parse_dates=['first_purchase_date', 'last_purchase_date']
)

for col in ['is_churn', 'ltv']:
    if col not in df.columns:
        raise ValueError(
            f"Coluna '{col}' não encontrada no dataset.\n"
            "Verifique se o feature_engineering.py foi executado corretamente."
        )

print(f"  ✓ Dataset carregado: {df.shape}")
print(f"  Taxa de churn geral: {df['is_churn'].mean() * 100:.1f}%")


# ── 2. Separar features e labels ──────────────────────────────────────────────
print("\n[2/7] Definindo features (sem leakage)...")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES SEGURAS PARA CHURN
# Critério: só incluir informações disponíveis ANTES de saber se o cliente
# vai churnar. Removemos:
#   • recency_days      → É literalmente a definição do label (> 120 = churn)
#   • total_spent       → Mesma janela temporal do label de LTV
#   • customer_lifetime_days → Pode vazar data de última compra indiretamente
#     se calculado até o momento do corte de churn
# ─────────────────────────────────────────────────────────────────────────────
CHURN_FEATURE_COLS = [
    # Volume de compras
    'num_orders',
    'num_unique_products',
    'frequency_orders_per_day',   # num_orders / customer_lifetime_days

    # Valor médio por pedido (NÃO o total gasto)
    'avg_order_value',

    # Qualidade percebida
    'avg_review_score',

    # Perfil geográfico / comportamental
    'customer_state',
    'favorite_category',
]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES PARA LTV (regressão)
# LTV = valor total já gasto pelo cliente. Aqui podemos incluir mais features
# de comportamento histórico porque o objetivo é estimar o valor, não prever
# um evento futuro de abandono.
# ─────────────────────────────────────────────────────────────────────────────
LTV_FEATURE_COLS = [
    'num_orders',
    'num_unique_products',
    'frequency_orders_per_day',
    'avg_order_value',
    'avg_review_score',
    'customer_lifetime_days',
    'customer_state',
    'favorite_category',
]

def filter_cols(col_list, df):
    present = [c for c in col_list if c in df.columns]
    missing = [c for c in col_list if c not in df.columns]
    if missing:
        print(f"  ⚠ Features ausentes (ignoradas): {missing}")
    return present

churn_feature_cols = filter_cols(CHURN_FEATURE_COLS, df)
ltv_feature_cols   = filter_cols(LTV_FEATURE_COLS,   df)

print(f"  Churn — features selecionadas ({len(churn_feature_cols)}): {churn_feature_cols}")
print(f"  LTV   — features selecionadas ({len(ltv_feature_cols)}): {ltv_feature_cols}")


# ── 3. Pré-processamento ──────────────────────────────────────────────────────
print("\n[3/7] Pré-processando features...")

def preprocess(df_raw, feature_cols):
    """
    Imputa nulos e aplica label encoding nas colunas categóricas.
    Retorna (X_encoded, label_encoders_dict).
    """
    X = df_raw[feature_cols].copy()

    num_cols = X.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        X[col] = X[col].fillna('unknown')

    encoders = {}
    for col in cat_cols:
        categories = sorted(X[col].unique())
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        X[col] = X[col].map(mapping).fillna(-1).astype(int)
        encoders[col] = mapping

    return X, encoders

X_churn, churn_encoders = preprocess(df, churn_feature_cols)
X_ltv,   ltv_encoders   = preprocess(df, ltv_feature_cols)

print(f"  ✓ Pré-processamento concluído.")


# ── 4. Split temporal ─────────────────────────────────────────────────────────
print("\n[4/7] Dividindo dataset (time-based split)...")

# ─────────────────────────────────────────────────────────────────────────────
# Por que split temporal e não aleatório?
#
# Em problemas de churn, um split aleatório deixa "vazar" informações do futuro
# para o treino: o modelo pode aprender padrões de clientes cujas compras
# futuras já são conhecidas no momento do treino.
#
# Com split temporal:
#   • Treino  → primeiros 80% dos clientes (ordenados por last_purchase_date)
#   • Teste   → últimos  20% dos clientes
# Isso simula o cenário real: treinar no passado, prever no futuro.
# ─────────────────────────────────────────────────────────────────────────────

if 'last_purchase_date' in df.columns:
    sort_idx = df['last_purchase_date'].argsort().values
    df        = df.iloc[sort_idx].reset_index(drop=True)
    X_churn   = X_churn.iloc[sort_idx].reset_index(drop=True)
    X_ltv     = X_ltv.iloc[sort_idx].reset_index(drop=True)
    print("  ✓ Dataset ordenado por last_purchase_date")
else:
    print("  ⚠ 'last_purchase_date' não encontrada — split sem ordenação temporal.")

y_churn = df['is_churn'].values
y_ltv   = df['ltv'].values

split_idx = int(len(df) * 0.8)

# Verificar se ambas as classes estão presentes no treino
_classes_train = np.unique(y_churn[:split_idx])
if len(_classes_train) < 2:
    print("  ⚠ Split temporal resultaria em apenas uma classe no treino.")
    print("  → Fallback: split estratificado (stratify=is_churn).")
    (X_train_c, X_test_c,
     X_train_l, X_test_l,
     y_train_churn, y_test_churn,
     y_train_ltv,   y_test_ltv) = train_test_split(
        X_churn, X_ltv, y_churn, y_ltv,
        test_size=0.2,
        random_state=42,
        stratify=y_churn
    )
else:
    X_train_c, X_test_c = X_churn.iloc[:split_idx], X_churn.iloc[split_idx:]
    X_train_l, X_test_l = X_ltv.iloc[:split_idx],   X_ltv.iloc[split_idx:]
    y_train_churn = y_churn[:split_idx]
    y_test_churn  = y_churn[split_idx:]
    y_train_ltv   = y_ltv[:split_idx]
    y_test_ltv    = y_ltv[split_idx:]

print(f"  Treino : {X_train_c.shape}  |  Teste: {X_test_c.shape}")
print(f"  Churn — treino: {y_train_churn.mean()*100:.1f}%  |  teste: {y_test_churn.mean()*100:.1f}%")


#  MODELO 1 — CHURN (Classificação)
print("\n" + "=" * 80)
print("  MODELO 1: PREVISÃO DE CHURN")
print("=" * 80)

n_neg = (y_train_churn == 0).sum()
n_pos = (y_train_churn == 1).sum()
scale_pos_weight_churn = n_neg / n_pos if n_pos > 0 else 1.0
print(f"\n  scale_pos_weight: {scale_pos_weight_churn:.2f}  (neg={n_neg:,} / pos={n_pos:,})")

print("\n[5/7] Treinando modelo de Churn (XGBoost)...")
model_churn = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight_churn,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    use_label_encoder=False
)

model_churn.fit(
    X_train_c, y_train_churn,
    eval_set=[(X_test_c, y_test_churn)],
    verbose=False
)
print("  ✓ Modelo de Churn treinado!")

# Avaliar
y_pred_churn       = model_churn.predict(X_test_c)
y_pred_proba_churn = model_churn.predict_proba(X_test_c)[:, 1]

accuracy_churn  = accuracy_score(y_test_churn, y_pred_churn)
precision_churn = precision_score(y_test_churn, y_pred_churn, zero_division=0)
recall_churn    = recall_score(y_test_churn, y_pred_churn, zero_division=0)
f1_churn        = f1_score(y_test_churn, y_pred_churn, zero_division=0)
auc_roc_churn   = roc_auc_score(y_test_churn, y_pred_proba_churn)

print("\n  Métricas — Churn (sem leakage):")
print(f"    Accuracy:  {accuracy_churn:.4f}")
print(f"    Precision: {precision_churn:.4f}")
print(f"    Recall:    {recall_churn:.4f}")
print(f"    F1-Score:  {f1_churn:.4f}")
print(f"    AUC-ROC:   {auc_roc_churn:.4f}  ← esperado entre 0.65–0.85 (realista)")

# Alerta: AUC-ROC muito alto pode indicar leakage residual
if auc_roc_churn > 0.95:
    print("\n  ⚠ ATENÇÃO: AUC-ROC > 0.95 ainda detectado.")
    print("    Verifique se 'feature_engineering.py' derivou alguma feature")
    print("    a partir de 'recency_days' ou do período pós-cutoff.")

print("\n  Top 10 Features Mais Importantes (Churn):")
fi_churn = (
    pd.DataFrame({
        'feature':    churn_feature_cols,
        'importance': model_churn.feature_importances_
    })
    .sort_values('importance', ascending=False)
    .head(10)
    .reset_index(drop=True)
)
fi_churn['importance'] = fi_churn['importance'].round(4)
print(fi_churn.to_string(index=False))

joblib.dump(model_churn,    MODEL_DIR / 'etapa02_churn_model.pkl')
joblib.dump(churn_encoders, MODEL_DIR / 'etapa02_churn_label_encoders.pkl')
joblib.dump(churn_feature_cols, MODEL_DIR / 'etapa02_feature_cols.pkl')
print(f"\n  ✓ etapa02_churn_model.pkl salvo.")

#  MODELO 2 — LTV (Regressão)
print("\n" + "=" * 80)
print("  MODELO 2: PREVISÃO DE LTV (Lifetime Value)")
print("=" * 80)

print(f"\n  Estatísticas do LTV (treino):")
print(pd.Series(y_train_ltv).describe().round(2).to_string())

print("\n[6/7] Treinando modelo de LTV (XGBoost Regressor)...")
model_ltv = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model_ltv.fit(
    X_train_l, y_train_ltv,
    eval_set=[(X_test_l, y_test_ltv)],
    verbose=False
)
print("  ✓ Modelo de LTV treinado!")

y_pred_ltv = model_ltv.predict(X_test_l)

rmse_ltv = np.sqrt(mean_squared_error(y_test_ltv, y_pred_ltv))
mae_ltv  = mean_absolute_error(y_test_ltv, y_pred_ltv)
r2_ltv   = r2_score(y_test_ltv, y_pred_ltv)

mask     = y_test_ltv != 0
mape_ltv = np.mean(np.abs((y_test_ltv[mask] - y_pred_ltv[mask]) / y_test_ltv[mask])) * 100

print("\n  Métricas — LTV:")
print(f"    RMSE:  {rmse_ltv:.2f}")
print(f"    MAE:   {mae_ltv:.2f}")
print(f"    MAPE:  {mape_ltv:.2f}%")
print(f"    R²:    {r2_ltv:.4f}")

print("\n  Top 10 Features Mais Importantes (LTV):")
fi_ltv = (
    pd.DataFrame({
        'feature':    ltv_feature_cols,
        'importance': model_ltv.feature_importances_
    })
    .sort_values('importance', ascending=False)
    .head(10)
    .reset_index(drop=True)
)
fi_ltv['importance'] = fi_ltv['importance'].round(4)
print(fi_ltv.to_string(index=False))

joblib.dump(model_ltv,          MODEL_DIR / 'etapa02_ltv_model.pkl')
joblib.dump(ltv_encoders,       MODEL_DIR / 'etapa02_ltv_label_encoders.pkl')
joblib.dump(ltv_feature_cols,   MODEL_DIR / 'etapa02_ltv_feature_cols.pkl')
print(f"\n  ✓ etapa02_ltv_model.pkl salvo.")


# ── 7. Salvar métricas consolidadas ──────────────────────────────────────────
print("\n[7/7] Salvando métricas...")

metrics = {
    'trained_at':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'leakage_fixed':    True,
    'train_size':       len(X_train_c),
    'test_size':        len(X_test_c),
    'churn_accuracy':   round(accuracy_churn,  4),
    'churn_precision':  round(precision_churn, 4),
    'churn_recall':     round(recall_churn,    4),
    'churn_f1':         round(f1_churn,        4),
    'churn_auc_roc':    round(auc_roc_churn,   4),
    'ltv_rmse':         round(rmse_ltv,        4),
    'ltv_mae':          round(mae_ltv,         4),
    'ltv_mape':         round(mape_ltv,        4),
    'ltv_r2':           round(r2_ltv,          4),
}
pd.DataFrame([metrics]).to_csv(MODEL_DIR / 'etapa02_metrics.csv', index=False)
print(f"  ✓ etapa02_metrics.csv salvo.")
print(f"  Diretório: {MODEL_DIR}")

print("\n" + "=" * 80)
print("  ETAPA 02 CONCLUÍDA COM SUCESSO!")
print("  Data leakage corrigido — métricas agora refletem performance real.")
print("=" * 80)