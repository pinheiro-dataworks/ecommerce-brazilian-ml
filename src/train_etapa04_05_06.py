"""
ETAPAS 04, 05, 06 - Treinamento dos Modelos Restantes
- Etapa 04: Recomendação (popularidade + mapeamentos para ALS)
- Etapa 05: Precificação Inteligente (XGBoost Regressor)
- Etapa 06: Clustering de Sellers (K-Means)

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    mean_squared_error, mean_absolute_error, r2_score
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

# Utilitários
def load_csv(path: Path, label: str, **kwargs) -> pd.DataFrame:
    """Carrega CSV com mensagem de erro clara se o arquivo não existir."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            "Execute feature_engineering.py antes deste script."
        )
    df = pd.read_csv(path, **kwargs)
    print(f"  ✓ {label}: {df.shape}")
    return df


# ETAPA 04 — SISTEMA DE RECOMENDAÇÃO
print("=" * 80)
print("  ETAPA 04 — SISTEMA DE RECOMENDAÇÃO")
print("=" * 80)

print("\n[1/3] Carregando dataset de interações...")
interactions = load_csv(DATA_DIR / 'etapa04_recommendation_dataset.csv', 'Interações')

# Criar mapeamentos de IDs (necessários para ALS ou inferência futura)
print("\n[2/3] Criando mapeamentos de IDs...")
customer_to_idx = {c: i for i, c in enumerate(interactions['customer_unique_id'].unique())}
product_to_idx  = {p: i for i, p in enumerate(interactions['product_id'].unique())}
# Inversos derivados dos dicts acima (conveniência para inferência)
idx_to_customer = {i: c for c, i in customer_to_idx.items()}
idx_to_product  = {i: p for p, i in product_to_idx.items()}

interactions['customer_idx'] = interactions['customer_unique_id'].map(customer_to_idx)
interactions['product_idx']  = interactions['product_id'].map(product_to_idx)

print(f"  Clientes únicos: {len(customer_to_idx):,}")
print(f"  Produtos únicos: {len(product_to_idx):,}")

# Modelo de popularidade (baseline — substitua por ALS com implicit/surprise)
product_popularity = (
    interactions.groupby('product_id')['interaction_count']
    .sum()
    .sort_values(ascending=False)
)
top_products = product_popularity.head(50).index.tolist()
print(f"  Top 50 produtos mais populares identificados.")

# Salvar artefatos
print("\n[3/3] Salvando artefatos de recomendação...")
joblib.dump(customer_to_idx, MODEL_DIR / 'etapa04_customer_to_idx.pkl')
joblib.dump(product_to_idx,  MODEL_DIR / 'etapa04_product_to_idx.pkl')
joblib.dump(idx_to_customer, MODEL_DIR / 'etapa04_idx_to_customer.pkl')
joblib.dump(idx_to_product,  MODEL_DIR / 'etapa04_idx_to_product.pkl')
joblib.dump(top_products,    MODEL_DIR / 'etapa04_top_products.pkl')

interactions[['customer_idx', 'product_idx', 'interaction_count']].to_csv(
    MODEL_DIR / 'etapa04_interaction_matrix.csv', index=False
)

# Métricas com timestamp
metrics_04 = {
    'trained_at':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'num_customers':   len(customer_to_idx),
    'num_products':    len(product_to_idx),
    'num_interactions':len(interactions),
    'top_n_products':  50,
}
pd.DataFrame([metrics_04]).to_csv(MODEL_DIR / 'etapa04_metrics.csv', index=False)

print(f"  ✓ etapa04_customer_to_idx.pkl")
print(f"  ✓ etapa04_product_to_idx.pkl")
print(f"  ✓ etapa04_idx_to_customer.pkl")
print(f"  ✓ etapa04_idx_to_product.pkl")
print(f"  ✓ etapa04_top_products.pkl")
print(f"  ✓ etapa04_interaction_matrix.csv")
print(f"  ✓ etapa04_metrics.csv")

print("\n" + "=" * 80)
print("  ETAPA 04 CONCLUÍDA!")
print("=" * 80)


# ETAPA 05 — PRECIFICAÇÃO INTELIGENTE (XGBoost)
print("\n" + "=" * 80)
print("  ETAPA 05 — PRECIFICAÇÃO INTELIGENTE")
print("=" * 80)

print("\n[1/5] Carregando dataset de precificação...")
df_pricing = load_csv(DATA_DIR / 'etapa05_pricing_dataset.csv', 'Precificação')

# Preparar features
print("\n[2/5] Preparando features...")
FEATURE_COLS_PRICING = [
    'product_weight_g', 'product_length_cm', 'product_height_cm',
    'product_width_cm', 'product_volume_cm3', 'product_photos_qty',
    'freight_value', 'purchase_year', 'purchase_month', 'purchase_quarter',
    'product_demand_count', 'category_avg_price',
    'product_category_name_english', 'seller_state'
]

feature_cols_pricing = [c for c in FEATURE_COLS_PRICING if c in df_pricing.columns]
missing_cols = [c for c in FEATURE_COLS_PRICING if c not in df_pricing.columns]
print(f"  Features selecionadas: {len(feature_cols_pricing)}")
if missing_cols:
    print(f"  ⚠ Features ausentes (ignoradas): {missing_cols}")

X_pricing = df_pricing[feature_cols_pricing].copy()
y_pricing  = df_pricing['price'].values

# Tratar nulos
num_cols_p = X_pricing.select_dtypes(include=['float64', 'int64']).columns
cat_cols_p = X_pricing.select_dtypes(include=['object']).columns

for col in num_cols_p:
    X_pricing[col] = X_pricing[col].fillna(X_pricing[col].median())
for col in cat_cols_p:
    X_pricing[col] = X_pricing[col].fillna('unknown')

# Encoding com mapeamento robusto (evita unseen labels no teste)
label_encoders_pricing = {}
for col in cat_cols_p:
    categories = sorted(X_pricing[col].unique())
    mapping = {cat: idx for idx, cat in enumerate(categories)}
    X_pricing[col] = X_pricing[col].map(mapping).fillna(-1).astype(int)
    label_encoders_pricing[col] = mapping

# Split temporal por purchase_year/purchase_month
print("\n[3/5] Dividindo dataset (time-based split)...")
if 'purchase_year' in df_pricing.columns and 'purchase_month' in df_pricing.columns:
    sort_idx = df_pricing[['purchase_year', 'purchase_month']].apply(
        lambda r: r['purchase_year'] * 100 + r['purchase_month'], axis=1
    ).argsort().values
    X_pricing = X_pricing.iloc[sort_idx].reset_index(drop=True)
    y_pricing  = y_pricing[sort_idx]
    print("  ✓ Dataset ordenado por purchase_year/purchase_month")
else:
    print("  ⚠ Colunas temporais não encontradas — split sem ordenação.")

split_idx_p = int(len(X_pricing) * 0.8)
X_train_p, X_test_p = X_pricing.iloc[:split_idx_p], X_pricing.iloc[split_idx_p:]
y_train_p, y_test_p = y_pricing[:split_idx_p], y_pricing[split_idx_p:]

print(f"  Treino: {X_train_p.shape}  |  Teste: {X_test_p.shape}")

# Treinar modelo
print("\n[4/5] Treinando modelo de precificação (XGBoost)...")
model_pricing = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model_pricing.fit(
    X_train_p, y_train_p,
    eval_set=[(X_test_p, y_test_p)],
    verbose=False
)
print("  ✓ Modelo treinado!")

# Avaliar
print("\n[5/5] Avaliando modelo...")
y_pred_p = model_pricing.predict(X_test_p)

rmse_p = np.sqrt(mean_squared_error(y_test_p, y_pred_p))
mae_p  = mean_absolute_error(y_test_p, y_pred_p)
r2_p   = r2_score(y_test_p, y_pred_p)

# MAPE com proteção contra divisão por zero
mask_p  = y_test_p != 0
mape_p  = np.mean(np.abs((y_test_p[mask_p] - y_pred_p[mask_p]) / y_test_p[mask_p])) * 100

print("\n  Métricas — Precificação:")
print(f"    RMSE:  R$ {rmse_p:.2f}")
print(f"    MAE:   R$ {mae_p:.2f}")
print(f"    MAPE:  {mape_p:.2f}%")
print(f"    R²:    {r2_p:.4f}")

print("\n  Top 10 Features Mais Importantes:")
fi_pricing = (
    pd.DataFrame({'feature': feature_cols_pricing, 'importance': model_pricing.feature_importances_})
    .sort_values('importance', ascending=False)
    .head(10)
    .reset_index(drop=True)
)
fi_pricing['importance'] = fi_pricing['importance'].round(4)
print(fi_pricing.to_string(index=False))

# Salvar artefatos
joblib.dump(model_pricing,          MODEL_DIR / 'etapa05_pricing_model.pkl')
joblib.dump(label_encoders_pricing, MODEL_DIR / 'etapa05_label_encoders.pkl')
joblib.dump(feature_cols_pricing,   MODEL_DIR / 'etapa05_feature_cols.pkl')

metrics_pricing = {
    'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'train_size': len(X_train_p),
    'test_size':  len(X_test_p),
    'rmse':       round(rmse_p, 4),
    'mae':        round(mae_p,  4),
    'mape':       round(mape_p, 4),
    'r2':         round(r2_p,   4),
}
pd.DataFrame([metrics_pricing]).to_csv(MODEL_DIR / 'etapa05_metrics.csv', index=False)

print(f"\n  ✓ etapa05_pricing_model.pkl")
print(f"  ✓ etapa05_label_encoders.pkl")
print(f"  ✓ etapa05_feature_cols.pkl")
print(f"  ✓ etapa05_metrics.csv")
print(f"  Diretório: {MODEL_DIR}")

print("\n" + "=" * 80)
print("  ETAPA 05 CONCLUÍDA!")
print("=" * 80)


# ETAPA 06 — CLUSTERING DE SELLERS (K-Means)
print("\n" + "=" * 80)
print("  ETAPA 06 — CLUSTERING DE SELLERS")
print("=" * 80)

print("\n[1/5] Carregando dataset de sellers...")
df_sellers = load_csv(DATA_DIR / 'etapa06_seller_clustering_dataset.csv', 'Sellers')

# Preparar features
print("\n[2/5] Preparando features...")
FEATURE_COLS_SELLERS = [
    'num_orders', 'total_revenue', 'avg_order_value', 'avg_freight',
    'num_unique_products', 'avg_review_score', 'num_states_served',
    'num_customers', 'delay_rate'
]

feature_cols_sellers = [c for c in FEATURE_COLS_SELLERS if c in df_sellers.columns]
missing_cols_s = [c for c in FEATURE_COLS_SELLERS if c not in df_sellers.columns]
print(f"  Features selecionadas: {len(feature_cols_sellers)}")
if missing_cols_s:
    print(f"  ⚠ Features ausentes (ignoradas): {missing_cols_s}")

X_sellers = df_sellers[feature_cols_sellers].copy()
X_sellers  = X_sellers.fillna(X_sellers.median())

# Normalizar features
print("\n[3/5] Normalizando features (StandardScaler)...")
scaler = StandardScaler()
X_sellers_scaled = scaler.fit_transform(X_sellers)

# Elbow Method — determinar K ótimo
print("\n[4/5] Determinando número ótimo de clusters (Elbow + Silhouette)...")
inertias           = []
silhouette_scores  = []
K_range            = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_sellers_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_sellers_scaled, km.labels_))

# Print tabular dos scores por K
print(f"\n  {'K':>4}  {'Inércia':>12}  {'Silhouette':>12}")
print(f"  {'-'*4}  {'-'*12}  {'-'*12}")
for k, inertia, sil in zip(K_range, inertias, silhouette_scores):
    print(f"  {k:>4}  {inertia:>12,.1f}  {sil:>12.4f}")

best_k   = list(K_range)[int(np.argmax(silhouette_scores))]
best_sil = max(silhouette_scores)
print(f"\n  ✓ K ótimo selecionado: {best_k}  (Silhouette = {best_sil:.4f})")

# Treinar modelo final
print(f"\n[5/5] Treinando K-Means com K={best_k}...")
model_clustering = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_sellers['cluster'] = model_clustering.fit_predict(X_sellers_scaled)

print("\n  Distribuição de Clusters:")
cluster_dist = df_sellers['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_dist.items():
    print(f"    Cluster {cluster_id}: {count:,} sellers  ({count / len(df_sellers) * 100:.1f}%)")

print("\n  Perfil Médio dos Clusters:")
cluster_summary = df_sellers.groupby('cluster')[feature_cols_sellers].mean().round(2)
print(cluster_summary.to_string())

# Salvar artefatos
# Nota: sellers_with_clusters é dado enriquecido → DATA_DIR (não MODEL_DIR)
joblib.dump(model_clustering,    MODEL_DIR / 'etapa06_clustering_model.pkl')
joblib.dump(scaler,              MODEL_DIR / 'etapa06_scaler.pkl')
joblib.dump(feature_cols_sellers,MODEL_DIR / 'etapa06_feature_cols.pkl')
df_sellers.to_csv(DATA_DIR / 'etapa06_sellers_with_clusters.csv', index=False)

metrics_clustering = {
    'trained_at':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'num_sellers':      len(df_sellers),
    'best_k':           best_k,
    'silhouette_score': round(best_sil, 4),
    'inertia':          round(model_clustering.inertia_, 2),
}
pd.DataFrame([metrics_clustering]).to_csv(MODEL_DIR / 'etapa06_metrics.csv', index=False)

print(f"\n  ✓ etapa06_clustering_model.pkl")
print(f"  ✓ etapa06_scaler.pkl")
print(f"  ✓ etapa06_feature_cols.pkl")
print(f"  ✓ etapa06_sellers_with_clusters.csv  → {DATA_DIR}")
print(f"  ✓ etapa06_metrics.csv")
print(f"  Diretório modelos: {MODEL_DIR}")

print("\n" + "=" * 80)
print("  ETAPA 06 CONCLUÍDA!")
print("=" * 80)

print("\n" + "=" * 80)
print("  ETAPAS 04, 05, 06 CONCLUÍDAS COM SUCESSO!")
print("=" * 80)