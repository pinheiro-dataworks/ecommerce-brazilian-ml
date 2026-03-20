"""
Script de Ingestão e ETL V2 - Projeto E-Commerce ML
Inclui geolocalização e cálculo de distância cliente-seller

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


# Função Haversine
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula distância haversine entre dois pontos em km.
    Aceita scalars ou Series do pandas (sem NaN).
    """
    R = 6371  # Raio da Terra em km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

# Caminhos
BASE_DIR    = Path(__file__).resolve().parent.parent  # raiz do projeto
DATA_DIR    = BASE_DIR / 'dataset'                    # CSVs originais do Olist
OUTPUT_DIR  = BASE_DIR / 'data'                       # saída processada

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("  FASE 1 - INGESTÃO E ETL V2 (com Geolocalização)")
print("=" * 80)


# Utilitário: carregamento seguro
def load_csv(path: Path, label: str) -> pd.DataFrame:
    """Carrega CSV com mensagem de erro clara se o arquivo não existir."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            f"Verifique se os CSVs do Olist estão em: {DATA_DIR}"
        )
    df = pd.read_csv(path)
    print(f"  ✓ {label}: {df.shape}")
    return df


# 1. Carregar todos os datasets
print("\n[1/7] Carregando datasets...")

orders               = load_csv(DATA_DIR / 'olist_orders_dataset.csv',                  'Orders')
order_items          = load_csv(DATA_DIR / 'olist_order_items_dataset.csv',              'Order Items')
payments             = load_csv(DATA_DIR / 'olist_order_payments_dataset.csv',           'Payments')
reviews              = load_csv(DATA_DIR / 'olist_order_reviews_dataset.csv',            'Reviews')
customers            = load_csv(DATA_DIR / 'olist_customers_dataset.csv',                'Customers')
sellers              = load_csv(DATA_DIR / 'olist_sellers_dataset.csv',                  'Sellers')
products             = load_csv(DATA_DIR / 'olist_products_dataset.csv',                 'Products')
geolocation          = load_csv(DATA_DIR / 'olist_geolocation_dataset.csv',              'Geolocation')
category_translation = load_csv(DATA_DIR / 'product_category_name_translation.csv',     'Category Translation')


# 2. Processar geolocalização (média por CEP)
print("\n[2/7] Processando geolocalização...")

def safe_mode(x):
    """Retorna a moda de uma Series; se vazia, retorna NaN."""
    m = x.mode()
    return m.iloc[0] if not m.empty else np.nan

geo_agg = (
    geolocation
    .groupby('geolocation_zip_code_prefix', as_index=False)
    .agg(
        geolocation_lat   = ('geolocation_lat',   'mean'),
        geolocation_lng   = ('geolocation_lng',   'mean'),
        geolocation_city  = ('geolocation_city',  safe_mode),
        geolocation_state = ('geolocation_state', safe_mode)
    )
)
print(f"  ✓ Geolocalização agregada por CEP: {geo_agg.shape}")


# 3. Join geolocalização com customers
print("\n[3/7] Adicionando geolocalização aos clientes...")

customers_geo = (
    customers
    .merge(
        geo_agg,
        left_on='customer_zip_code_prefix',
        right_on='geolocation_zip_code_prefix',
        how='left'
    )
    .rename(columns={
        'geolocation_lat':   'customer_lat',
        'geolocation_lng':   'customer_lng',
        'geolocation_city':  'customer_geo_city',
        'geolocation_state': 'customer_geo_state'
    })
    .drop(columns=['geolocation_zip_code_prefix'])
)
print(f"  ✓ Customers com geolocalização: {customers_geo.shape}")


# 4. Join geolocalização com sellers
print("\n[4/7] Adicionando geolocalização aos vendedores...")

sellers_geo = (
    sellers
    .merge(
        geo_agg,
        left_on='seller_zip_code_prefix',
        right_on='geolocation_zip_code_prefix',
        how='left'
    )
    .rename(columns={
        'geolocation_lat':   'seller_lat',
        'geolocation_lng':   'seller_lng',
        'geolocation_city':  'seller_geo_city',
        'geolocation_state': 'seller_geo_state'
    })
    .drop(columns=['geolocation_zip_code_prefix'])
)
print(f"  ✓ Sellers com geolocalização: {sellers_geo.shape}")


# 5. Converter datas
print("\n[5/7] Convertendo colunas de data...")

date_cols_orders = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]
for col in date_cols_orders:
    if col in orders.columns:
        orders[col] = pd.to_datetime(orders[col], errors='coerce')

for col in ['review_creation_date', 'review_answer_timestamp']:
    if col in reviews.columns:
        reviews[col] = pd.to_datetime(reviews[col], errors='coerce')

print("  ✓ Datas convertidas com sucesso.")


# 6. Criar master_table através de joins
print("\n[6/7] Criando master_table unificada com geolocalização...")

master = orders.copy()
print(f"  Base (orders):                      {master.shape}")

master = master.merge(customers_geo, on='customer_id', how='left')
print(f"  Após join com customers (geo):      {master.shape}")

master = master.merge(order_items, on='order_id', how='left')
print(f"  Após join com order_items:          {master.shape}")

# Selecionar apenas colunas necessárias de sellers_geo para evitar ambiguidade
sellers_cols = ['seller_id', 'seller_zip_code_prefix', 'seller_city',
                'seller_state', 'seller_lat', 'seller_lng',
                'seller_geo_city', 'seller_geo_state']
sellers_cols_existing = [c for c in sellers_cols if c in sellers_geo.columns]

master = master.merge(sellers_geo[sellers_cols_existing], on='seller_id', how='left')
print(f"  Após join com sellers (geo):        {master.shape}")

master = master.merge(products, on='product_id', how='left')
print(f"  Após join com products:             {master.shape}")

master = master.merge(category_translation, on='product_category_name', how='left')
print(f"  Após join com category_translation: {master.shape}")

# --- Agregar payments por order_id ---
payments_agg = (
    payments
    .groupby('order_id')
    .agg(
        payment_count            = ('payment_sequential', 'count'),
        payment_type_main        = ('payment_type',        safe_mode),
        payment_installments_max = ('payment_installments', 'max'),
        payment_value_total      = ('payment_value',        'sum')
    )
    .reset_index()
)

master = master.merge(payments_agg, on='order_id', how='left')
print(f"  Após join com payments (agregado):  {master.shape}")

# --- Agregar reviews por order_id (review mais recente por answer_timestamp) ---
sort_col = 'review_answer_timestamp' if 'review_answer_timestamp' in reviews.columns \
           else 'review_creation_date'

reviews_agg = (
    reviews
    .sort_values(sort_col, na_position='first')
    .groupby('order_id')
    .agg(
        review_score          = ('review_score',           'last'),   # score do review mais recente
        review_score_mean     = ('review_score',           'mean'),   # média de todos os reviews
        review_title          = ('review_comment_title',   'last'),   # título do mais recente
        review_message        = ('review_comment_message', 'last')    # mensagem do mais recente
    )
    .reset_index()
)

master = master.merge(reviews_agg, on='order_id', how='left')
print(f"  Após join com reviews (agregado):   {master.shape}")

# 7. Calcular distância haversine cliente-seller
print("\n[7/7] Calculando distância cliente-seller...")

mask_valid = (
    master['customer_lat'].notna() &
    master['customer_lng'].notna() &
    master['seller_lat'].notna()   &
    master['seller_lng'].notna()
)

master['distance_customer_seller_km'] = np.nan

master.loc[mask_valid, 'distance_customer_seller_km'] = haversine_distance(
    master.loc[mask_valid, 'customer_lat'],
    master.loc[mask_valid, 'customer_lng'],
    master.loc[mask_valid, 'seller_lat'],
    master.loc[mask_valid, 'seller_lng']
)

n_calculados = mask_valid.sum()
pct = n_calculados / len(master) * 100
print(f"  ✓ Distância calculada para {n_calculados:,} registros ({pct:.1f}%)")
print(f"  ✓ Master table criada: {master.shape} | {len(master.columns)} colunas")

print("\n  Estatísticas de Distância Cliente-Seller (km):")
print(master['distance_customer_seller_km'].describe().round(2).to_string())

# Salvar arquivos
print(f"\nSalvando arquivos em: {OUTPUT_DIR}")

master.to_csv(         OUTPUT_DIR / 'master_table_v2.csv',      index=False)
print("  ✓ master_table_v2.csv")

orders.to_csv(         OUTPUT_DIR / 'orders_processed.csv',     index=False)
print("  ✓ orders_processed.csv")

customers_geo.to_csv(  OUTPUT_DIR / 'customers_geo.csv',        index=False)
print("  ✓ customers_geo.csv")

sellers_geo.to_csv(    OUTPUT_DIR / 'sellers_geo.csv',          index=False)
print("  ✓ sellers_geo.csv")

products.to_csv(       OUTPUT_DIR / 'products_processed.csv',   index=False)
print("  ✓ products_processed.csv")

geo_agg.to_csv(        OUTPUT_DIR / 'geolocation_agg.csv',      index=False)
print("  ✓ geolocation_agg.csv")

reviews.to_csv(        OUTPUT_DIR / 'reviews_processed.csv',    index=False)
print("  ✓ reviews_processed.csv")

print("\n" + "=" * 80)
print("  FASE 1 V2 CONCLUÍDA COM SUCESSO!")
print("  Master table com geolocalização e distância criada!")
print("=" * 80)