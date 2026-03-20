"""
FASE 3 - Feature Engineering para as 6 Etapas de ML
Cria datasets específicos para cada modelo

"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configurações do Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)


# Caminhos portáteis
BASE_DIR   = Path(__file__).resolve().parent.parent  # raiz do projeto
DATA_DIR   = BASE_DIR / 'data'
OUTPUT_DIR = DATA_DIR

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Pasta 'data/' não encontrada em: {BASE_DIR}\n"
        "Execute data_ingestion_v2.py antes deste script."
    )

print("=" * 80)
print("  FASE 3 - FEATURE ENGINEERING")
print("=" * 80)


# Utilitários
def safe_mode(x):
    """Retorna a moda de uma Series; se vazia, retorna NaN."""
    m = x.mode()
    return m.iloc[0] if not m.empty else np.nan


def load_csv(path: Path, label: str, **kwargs) -> pd.DataFrame:
    """Carrega CSV com mensagem de erro clara se o arquivo não existir."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            f"Execute os scripts anteriores antes deste."
        )
    df = pd.read_csv(path, **kwargs)
    print(f"  ✓ {label}: {df.shape}")
    return df

# 1. Carregar master_table v2
print("\n[1/7] Carregando master_table v2...")

date_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

master = load_csv(
    DATA_DIR / 'master_table_v2.csv',
    'Master Table v2',
    parse_dates=date_cols
)

# Detectar coluna de review score disponível
review_score_col = (
    'review_score'      if 'review_score'      in master.columns else
    'review_score_mean' if 'review_score_mean' in master.columns else
    None
)
if review_score_col:
    print(f"  ✓ Coluna de review score detectada: '{review_score_col}'")
else:
    print("  ⚠ Nenhuma coluna de review score encontrada.")


# ETAPA 01 — PREVISÃO DE ATRASO NA ENTREGA (Classificação Binária)
print("\n" + "=" * 80)
print("  [2/7] ETAPA 01 — PREVISÃO DE ATRASO NA ENTREGA")
print("=" * 80)

df_delay = master[
    (master['order_status'] == 'delivered') &
    master['order_delivered_customer_date'].notna() &
    master['order_estimated_delivery_date'].notna()
].copy()

print(f"  Registros válidos para análise de atraso: {df_delay.shape}")

# Colunas condicionais (dependem da versão da master)
agg_dict = {
    'order_purchase_timestamp':    'first',
    'order_approved_at':           'first',
    'order_delivered_carrier_date':'first',
    'order_delivered_customer_date':'first',
    'order_estimated_delivery_date':'first',
    'customer_id':                 'first',
    'customer_state':              'first',
    'customer_city':               'first',
    'customer_zip_code_prefix':    'first',
    'order_item_id':               'count',
    'price':                       'sum',
    'freight_value':               'sum',
    'seller_id':                   safe_mode,
    'seller_state':                safe_mode,
    'product_id':                  safe_mode,
    'product_category_name_english': safe_mode,
    'product_weight_g':            'sum',
    'product_length_cm':           'mean',
    'product_height_cm':           'mean',
    'product_width_cm':            'mean',
    'product_photos_qty':          'mean',
    'payment_value_total':         'first',
    'payment_type_main':           'first',
    'payment_installments_max':    'first',
}

# Adicionar colunas opcionais se existirem
for col in ['customer_lat', 'customer_lng', 'seller_lat', 'seller_lng',
            'distance_customer_seller_km']:
    if col in df_delay.columns:
        agg_dict[col] = 'mean' if 'lat' in col or 'lng' in col or 'km' in col else 'first'

# Filtrar apenas colunas existentes
agg_dict = {k: v for k, v in agg_dict.items() if k in df_delay.columns}

delay_features = df_delay.groupby('order_id').agg(agg_dict).reset_index()
print(f"  Features agregadas por pedido: {delay_features.shape}")

# Target: atraso (1) ou no prazo (0)
delay_features['delivery_delay_days'] = (
    delay_features['order_delivered_customer_date'] -
    delay_features['order_estimated_delivery_date']
).dt.total_seconds() / (24 * 3600)

delay_features['is_delayed'] = (delay_features['delivery_delay_days'] > 0).astype(int)

# Features temporais (sem data leakage — apenas da compra)
ts = delay_features['order_purchase_timestamp']
delay_features['purchase_year']     = ts.dt.year
delay_features['purchase_month']    = ts.dt.month
delay_features['purchase_day']      = ts.dt.day
delay_features['purchase_weekday']  = ts.dt.weekday
delay_features['purchase_hour']     = ts.dt.hour
delay_features['purchase_quarter']  = ts.dt.quarter

# Prazo prometido (dias entre compra e entrega estimada)
delay_features['promised_delivery_days'] = (
    delay_features['order_estimated_delivery_date'] -
    delay_features['order_purchase_timestamp']
).dt.total_seconds() / (24 * 3600)

# Volume do produto (cm³)
if all(c in delay_features.columns for c in ['product_length_cm', 'product_height_cm', 'product_width_cm']):
    delay_features['product_volume_cm3'] = (
        delay_features['product_length_cm'] *
        delay_features['product_height_cm'] *
        delay_features['product_width_cm']
    )

# Mesma UF cliente-seller?
if 'seller_state' in delay_features.columns:
    delay_features['same_state'] = (
        delay_features['customer_state'] == delay_features['seller_state']
    ).astype(int)

n_total   = len(delay_features)
n_delayed = delay_features['is_delayed'].sum()
print(f"\n  Target Distribution:")
print(f"    Total de pedidos:  {n_total:,}")
print(f"    Atrasados:         {n_delayed:,}  ({n_delayed / n_total * 100:.1f}%)")
print(f"    No prazo:          {n_total - n_delayed:,}  ({(n_total - n_delayed) / n_total * 100:.1f}%)")

delay_features.to_csv(OUTPUT_DIR / 'etapa01_delay_dataset.csv', index=False)
print(f"\n  ✓ etapa01_delay_dataset.csv salvo.")

# ETAPA 02 — CHURN E LTV (Classificação + Regressão)
print("\n" + "=" * 80)
print("  [3/7] ETAPA 02 — CHURN E LTV")
print("=" * 80)

ref_date = master['order_purchase_timestamp'].max()
print(f"  Data de referência: {ref_date.date()}")

# Usar pedidos únicos para evitar duplicidade de payment_value_total
master_orders_unique = master.drop_duplicates('order_id')

churn_agg = {
    'order_id':                    'nunique',
    'order_purchase_timestamp':    ['min', 'max'],
    'payment_value_total':         'sum',
    'customer_state':              'first',
    'customer_city':               'first',
    'customer_zip_code_prefix':    'first',
    'product_id':                  'nunique',
}

# Adicionar category e review_score se existirem
if 'product_category_name_english' in master.columns:
    churn_agg['product_category_name_english'] = safe_mode
if review_score_col:
    churn_agg[review_score_col] = 'mean'

churn_agg = {k: v for k, v in churn_agg.items() if k in master.columns}

churn_features = master.groupby('customer_unique_id').agg(churn_agg).reset_index()

# Achatar MultiIndex de colunas
churn_features.columns = [
    '_'.join(filter(None, col)).strip('_') if isinstance(col, tuple) else col
    for col in churn_features.columns
]

# Renomear colunas para nomes limpos
rename_map = {
    'customer_unique_id':                    'customer_unique_id',
    'order_id_nunique':                      'num_orders',
    'order_purchase_timestamp_min':          'first_purchase_date',
    'order_purchase_timestamp_max':          'last_purchase_date',
    'payment_value_total_sum':               'total_spent',
    'customer_state_first':                  'customer_state',
    'customer_city_first':                   'customer_city',
    'customer_zip_code_prefix_first':        'customer_zip_code_prefix',
    'product_id_nunique':                    'num_unique_products',
    'product_category_name_english_<lambda>':'favorite_category',
    'product_category_name_english_safe_mode':'favorite_category',
}
if review_score_col:
    rename_map[f'{review_score_col}_mean'] = 'avg_review_score'

churn_features = churn_features.rename(columns={
    k: v for k, v in rename_map.items() if k in churn_features.columns
})

# Converter datas
for col in ['first_purchase_date', 'last_purchase_date']:
    if col in churn_features.columns:
        churn_features[col] = pd.to_datetime(churn_features[col])

# Features derivadas
churn_features['recency_days'] = (
    ref_date - churn_features['last_purchase_date']
).dt.total_seconds() / (24 * 3600)

churn_features['customer_lifetime_days'] = (
    churn_features['last_purchase_date'] - churn_features['first_purchase_date']
).dt.total_seconds() / (24 * 3600)

churn_features['frequency_orders_per_day'] = (
    churn_features['num_orders'] / (churn_features['customer_lifetime_days'] + 1)
)

churn_features['avg_order_value'] = (
    churn_features['total_spent'] / churn_features['num_orders']
)

# Targets
CHURN_THRESHOLD = 120
churn_features['is_churn'] = (churn_features['recency_days'] > CHURN_THRESHOLD).astype(int)
churn_features['ltv']      = churn_features['total_spent']

n_total = len(churn_features)
n_churn = churn_features['is_churn'].sum()
print(f"\n  Churn Distribution (threshold: {CHURN_THRESHOLD} dias):")
print(f"    Total de clientes: {n_total:,}")
print(f"    Churnados:         {n_churn:,}  ({n_churn / n_total * 100:.1f}%)")
print(f"    Ativos:            {n_total - n_churn:,}  ({(n_total - n_churn) / n_total * 100:.1f}%)")
print(f"\n  LTV Statistics:")
print(churn_features['ltv'].describe().round(2).to_string())

churn_features.to_csv(OUTPUT_DIR / 'etapa02_churn_ltv_dataset.csv', index=False)
print(f"\n  ✓ etapa02_churn_ltv_dataset.csv salvo.")


# ETAPA 03 — ANÁLISE DE SENTIMENTO (NLP)
print("\n" + "=" * 80)
print("  [4/7] ETAPA 03 — ANÁLISE DE SENTIMENTO")
print("=" * 80)

reviews = load_csv(DATA_DIR / 'reviews_processed.csv', 'Reviews')

# Filtrar apenas reviews com comentário
sentiment_df = reviews[reviews['review_comment_message'].notna()].copy()
print(f"  Reviews com comentários: {len(sentiment_df):,}")

# Target de sentimento: Negativo (1-2), Neutro (3), Positivo (4-5)
sentiment_df['sentiment'] = sentiment_df['review_score'].apply(
    lambda x: 'negative' if x <= 2 else ('neutral' if x == 3 else 'positive')
)

dist = sentiment_df['sentiment'].value_counts()
total = len(sentiment_df)
print(f"\n  Sentiment Distribution:")
for label, count in dist.items():
    print(f"    {label:<10} {count:>7,}  ({count / total * 100:.1f}%)")

# Colunas para NLP
nlp_cols = ['review_id', 'order_id', 'review_score', 'review_comment_title',
            'review_comment_message', 'sentiment']
nlp_cols_existing = [c for c in nlp_cols if c in sentiment_df.columns]

sentiment_df[nlp_cols_existing].to_csv(
    OUTPUT_DIR / 'etapa03_sentiment_dataset.csv', index=False
)
print(f"\n  ✓ etapa03_sentiment_dataset.csv salvo.")

# ETAPA 04 — SISTEMA DE RECOMENDAÇÃO
print("\n" + "=" * 80)
print("  [5/7] ETAPA 04 — SISTEMA DE RECOMENDAÇÃO")
print("=" * 80)

interactions = (
    master[['customer_unique_id', 'product_id', 'order_id']]
    .dropna()
    .groupby(['customer_unique_id', 'product_id'])
    .agg(interaction_count=('order_id', 'count'))
    .reset_index()
)

print(f"  Interações únicas:  {len(interactions):,}")
print(f"  Clientes únicos:    {interactions['customer_unique_id'].nunique():,}")
print(f"  Produtos únicos:    {interactions['product_id'].nunique():,}")

interactions.to_csv(OUTPUT_DIR / 'etapa04_recommendation_dataset.csv', index=False)
print(f"\n  ✓ etapa04_recommendation_dataset.csv salvo.")

# ETAPA 05 — PRECIFICAÇÃO INTELIGENTE (Elasticidade)
print("\n" + "=" * 80)
print("  [6/7] ETAPA 05 — PRECIFICAÇÃO INTELIGENTE")
print("=" * 80)

pricing_cols = [
    'order_id', 'order_item_id', 'product_id', 'seller_id',
    'price', 'freight_value', 'product_category_name_english',
    'product_weight_g', 'product_length_cm', 'product_height_cm',
    'product_width_cm', 'product_photos_qty', 'seller_state',
    'order_purchase_timestamp'
]
pricing_cols_existing = [c for c in pricing_cols if c in master.columns]

pricing_df = (
    master[pricing_cols_existing]
    .dropna(subset=['price', 'product_id'])
    .copy()
)

# Volume do produto (cm³)
if all(c in pricing_df.columns for c in ['product_length_cm', 'product_height_cm', 'product_width_cm']):
    pricing_df['product_volume_cm3'] = (
        pricing_df['product_length_cm'] *
        pricing_df['product_height_cm'] *
        pricing_df['product_width_cm']
    )

# Features temporais — coluna já é datetime, sem pd.to_datetime() redundante
ts = pricing_df['order_purchase_timestamp']
pricing_df['purchase_year']    = ts.dt.year
pricing_df['purchase_month']   = ts.dt.month
pricing_df['purchase_quarter'] = ts.dt.quarter

# Demanda histórica por produto
product_demand = (
    pricing_df.groupby('product_id')
    .size()
    .reset_index(name='product_demand_count')
)
pricing_df = pricing_df.merge(product_demand, on='product_id', how='left')

# Preço médio por categoria
if 'product_category_name_english' in pricing_df.columns:
    category_avg_price = (
        pricing_df.groupby('product_category_name_english')['price']
        .mean()
        .reset_index(name='category_avg_price')
    )
    pricing_df = pricing_df.merge(category_avg_price, on='product_category_name_english', how='left')

print(f"  Registros para precificação: {len(pricing_df):,}")
print(f"\n  Price Statistics:")
print(pricing_df['price'].describe().round(2).to_string())

pricing_df.to_csv(OUTPUT_DIR / 'etapa05_pricing_dataset.csv', index=False)
print(f"\n  ✓ etapa05_pricing_dataset.csv salvo.")


# ETAPA 06 — CLUSTERING DE SELLERS
print("\n" + "=" * 80)
print("  [7/7] ETAPA 06 — CLUSTERING DE SELLERS")
print("=" * 80)

seller_agg = {
    'order_id':    'nunique',
    'price':       ['sum', 'mean'],
    'freight_value': 'mean',
    'product_id':  'nunique',
    'customer_state': 'nunique',
    'customer_id': 'nunique',
}

# Colunas opcionais
if 'product_category_name_english' in master.columns:
    seller_agg['product_category_name_english'] = safe_mode
if review_score_col:
    seller_agg[review_score_col] = 'mean'
if 'seller_state' in master.columns:
    seller_agg['seller_state'] = 'first'
# Usar mean para coordenadas (evita NaN do 'first')
for col in ['seller_lat', 'seller_lng']:
    if col in master.columns:
        seller_agg[col] = 'mean'

seller_agg = {k: v for k, v in seller_agg.items() if k in master.columns}

seller_features = master.groupby('seller_id').agg(seller_agg).reset_index()

# Achatar MultiIndex de colunas
seller_features.columns = [
    '_'.join(filter(None, col)).strip('_') if isinstance(col, tuple) else col
    for col in seller_features.columns
]

# Renomear para nomes limpos
seller_rename = {
    'order_id_nunique':                       'num_orders',
    'price_sum':                              'total_revenue',
    'price_mean':                             'avg_order_value',
    'freight_value_mean':                     'avg_freight',
    'product_id_nunique':                     'num_unique_products',
    'product_category_name_english_safe_mode':'main_category',
    'customer_state_nunique':                 'num_states_served',
    'customer_id_nunique':                    'num_customers',
    'seller_state_first':                     'seller_state',
    'seller_lat_mean':                        'seller_lat',
    'seller_lng_mean':                        'seller_lng',
}
if review_score_col:
    seller_rename[f'{review_score_col}_mean'] = 'avg_review_score'

seller_features = seller_features.rename(columns={
    k: v for k, v in seller_rename.items() if k in seller_features.columns
})

# Taxa de atraso por seller (join com delay_features)
if 'seller_id' in delay_features.columns:
    seller_delay = (
        delay_features.groupby('seller_id')['is_delayed']
        .mean()
        .reset_index(name='delay_rate')
    )
    seller_features = seller_features.merge(seller_delay, on='seller_id', how='left')
    seller_features['delay_rate'] = seller_features['delay_rate'].fillna(0)

print(f"  Sellers para clustering: {len(seller_features):,}")
cols_to_describe = [c for c in ['num_orders', 'total_revenue', 'avg_review_score', 'delay_rate']
                    if c in seller_features.columns]
print(f"\n  Seller Statistics:")
print(seller_features[cols_to_describe].describe().round(2).to_string())

seller_features.to_csv(OUTPUT_DIR / 'etapa06_seller_clustering_dataset.csv', index=False)
print(f"\n  ✓ etapa06_seller_clustering_dataset.csv salvo.")


print("\n" + "=" * 80)
print("  FASE 3 - FEATURE ENGINEERING CONCLUÍDA COM SUCESSO!")
print("  6 datasets criados e salvos em: data/")
print("=" * 80)
