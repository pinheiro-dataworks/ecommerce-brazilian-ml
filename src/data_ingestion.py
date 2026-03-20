"""
Script de Ingestão e ETL - Projeto E-Commerce ML
Carrega todos os datasets e cria a master_table unificada

Estrutura esperada:
    PROJECT_01_ECOMMERCE_BR/
    ├── dataset/          ← CSVs originais do Olist
    ├── data/             ← saída processada (criada automaticamente)
    └── src/
        └── data_ingestion.py  ← este script
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Configurações do Pandas
# ============================================================
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)


# ============================================================
# Caminhos portáteis
# Script em: PROJECT_01_ECOMMERCE_BR/src/data_ingestion.py
# ============================================================
BASE_DIR   = Path(__file__).resolve().parent.parent  # raiz do projeto
DATA_DIR   = BASE_DIR / 'dataset'                    # CSVs originais do Olist
OUTPUT_DIR = BASE_DIR / 'data'                       # saída processada

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("  FASE 1 - INGESTÃO E ETL")
print("=" * 80)


# ============================================================
# Utilitários
# ============================================================
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


def safe_mode(x):
    """Retorna a moda de uma Series; se vazia, retorna NaN."""
    m = x.mode()
    return m.iloc[0] if not m.empty else np.nan


# ============================================================
# 1. Carregar todos os datasets
# ============================================================
print("\n[1/5] Carregando datasets...")

orders               = load_csv(DATA_DIR / 'olist_orders_dataset.csv',              'Orders')
order_items          = load_csv(DATA_DIR / 'olist_order_items_dataset.csv',          'Order Items')
payments             = load_csv(DATA_DIR / 'olist_order_payments_dataset.csv',       'Payments')
reviews              = load_csv(DATA_DIR / 'olist_order_reviews_dataset.csv',        'Reviews')
customers            = load_csv(DATA_DIR / 'olist_customers_dataset.csv',            'Customers')
sellers              = load_csv(DATA_DIR / 'olist_sellers_dataset.csv',              'Sellers')
products             = load_csv(DATA_DIR / 'olist_products_dataset.csv',             'Products')
geolocation          = load_csv(DATA_DIR / 'olist_geolocation_dataset.csv',          'Geolocation')
category_translation = load_csv(DATA_DIR / 'product_category_name_translation.csv', 'Category Translation')


# ============================================================
# 2. Converter datas
# ============================================================
print("\n[2/5] Convertendo colunas de data...")

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
        print(f"  ✓ {col}")

for col in ['review_creation_date', 'review_answer_timestamp']:
    if col in reviews.columns:
        reviews[col] = pd.to_datetime(reviews[col], errors='coerce')
        print(f"  ✓ {col}")


# ============================================================
# 3. Criar master_table através de joins
# ============================================================
print("\n[3/5] Criando master_table unificada...")

master = orders.copy()
print(f"  Base (orders):                      {master.shape}")

master = master.merge(customers, on='customer_id', how='left')
print(f"  Após join com customers:            {master.shape}")

master = master.merge(order_items, on='order_id', how='left')
print(f"  Após join com order_items:          {master.shape}")

# Selecionar colunas explícitas de sellers para evitar ambiguidade
sellers_cols = ['seller_id', 'seller_zip_code_prefix', 'seller_city', 'seller_state']
sellers_cols_existing = [c for c in sellers_cols if c in sellers.columns]

master = master.merge(sellers[sellers_cols_existing], on='seller_id', how='left')
print(f"  Após join com sellers:              {master.shape}")

master = master.merge(products, on='product_id', how='left')
print(f"  Após join com products:             {master.shape}")

master = master.merge(category_translation, on='product_category_name', how='left')
print(f"  Após join com category_translation: {master.shape}")

# --- Agregar payments por order_id ---
payments_agg = (
    payments
    .groupby('order_id')
    .agg(
        payment_count            = ('payment_sequential',  'count'),
        payment_type_main        = ('payment_type',         safe_mode),
        payment_installments_max = ('payment_installments', 'max'),
        payment_value_total      = ('payment_value',        'sum')
    )
    .reset_index()
)

master = master.merge(payments_agg, on='order_id', how='left')
print(f"  Após join com payments (agregado):  {master.shape}")

# --- Agregar reviews por order_id (review mais recente) ---
sort_col = 'review_answer_timestamp' if 'review_answer_timestamp' in reviews.columns \
           else 'review_creation_date'

reviews_agg = (
    reviews
    .sort_values(sort_col, na_position='first')
    .groupby('order_id')
    .agg(
        review_score      = ('review_score',           'last'),   # score do mais recente
        review_score_mean = ('review_score',           'mean'),   # média de todos
        review_title      = ('review_comment_title',   'last'),   # título do mais recente
        review_message    = ('review_comment_message', 'last')    # mensagem do mais recente
    )
    .reset_index()
)

master = master.merge(reviews_agg, on='order_id', how='left')
print(f"  Após join com reviews (agregado):   {master.shape}")

print(f"\n  ✓ Master table criada: {master.shape} | {len(master.columns)} colunas")


# ============================================================
# 4. Diagnóstico de qualidade
# ============================================================
print("\n[4/5] Diagnóstico de qualidade dos dados...")

null_counts = master.isnull().sum().sort_values(ascending=False)
null_counts = null_counts[null_counts > 0].head(20)
if not null_counts.empty:
    print("\n  Nulos por coluna (top 20):")
    print(null_counts.to_string())
else:
    print("  ✓ Nenhum valor nulo encontrado.")

duplicatas = master.duplicated().sum()
print(f"\n  Duplicatas encontradas: {duplicatas:,}")

print("\n  Estatísticas gerais:")
print(f"  • Registros totais:      {len(master):,}")
print(f"  • Pedidos únicos:        {master['order_id'].nunique():,}")
print(f"  • Clientes únicos:       {master['customer_id'].nunique():,}")
print(f"  • Vendedores únicos:     {master['seller_id'].nunique():,}")
print(f"  • Produtos únicos:       {master['product_id'].nunique():,}")
print(f"  • Período:               "
      f"{master['order_purchase_timestamp'].min()} "
      f"até {master['order_purchase_timestamp'].max()}")


# ============================================================
# 5. Salvar arquivos
# ============================================================
print(f"\n[5/5] Salvando arquivos em: {OUTPUT_DIR}")

master.to_csv(      OUTPUT_DIR / 'master_table.csv',          index=False)
print("  ✓ master_table.csv")

orders.to_csv(      OUTPUT_DIR / 'orders_processed.csv',      index=False)
print("  ✓ orders_processed.csv")

customers.to_csv(   OUTPUT_DIR / 'customers_processed.csv',   index=False)
print("  ✓ customers_processed.csv")

sellers.to_csv(     OUTPUT_DIR / 'sellers_processed.csv',     index=False)
print("  ✓ sellers_processed.csv")

products.to_csv(    OUTPUT_DIR / 'products_processed.csv',    index=False)
print("  ✓ products_processed.csv")

geolocation.to_csv( OUTPUT_DIR / 'geolocation_processed.csv', index=False)
print("  ✓ geolocation_processed.csv")

reviews.to_csv(     OUTPUT_DIR / 'reviews_processed.csv',     index=False)
print("  ✓ reviews_processed.csv")

print("\n" + "=" * 80)
print("  FASE 1 CONCLUÍDA COM SUCESSO!")
print("=" * 80)