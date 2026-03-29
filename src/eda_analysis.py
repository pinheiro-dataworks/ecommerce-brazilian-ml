"""
EDA - Análise Exploratória de Dados
Análise completa da master_table
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# === Configurações do Pandas ===

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

# === Caminhos portáteis ===

BASE_DIR   = Path(__file__).resolve().parent.parent  # raiz do projeto
DATA_DIR   = BASE_DIR / 'data'                       # entrada e saída

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Pasta 'data/' não encontrada em: {BASE_DIR}\n"
        "Execute data_ingestion.py antes deste script."
    )

print("=" * 80)
print("  FASE 2 - ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
print("=" * 80)

# === 1. Carregar master_table ===

print("\n[1/9] Carregando master_table...")

master_path = DATA_DIR / 'master_table_v2.csv'
if not master_path.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {master_path}\n"
        "Execute data_ingestion_v2.py antes deste script."
    )

master = pd.read_csv(
    master_path,
    parse_dates=[
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
)
print(f"  ✓ Master table carregada: {master.shape} | {len(master.columns)} colunas")

# === 2. Visão Geral ===

print("\n[2/9] Análise Descritiva Geral...")

print("\n  Estatísticas numéricas (colunas principais):")
num_cols = ['price', 'freight_value', 'payment_value_total', 'review_score',
            'distance_customer_seller_km']
num_cols_existing = [c for c in num_cols if c in master.columns]
print(master[num_cols_existing].describe().round(2).to_string())

print("\n  Top 10 colunas com valores nulos (%):")
null_pct = (master.isnull().sum() / len(master) * 100).sort_values(ascending=False).head(10)
null_pct = null_pct[null_pct > 0]
if not null_pct.empty:
    print(null_pct.round(2).to_string())
else:
    print("  ✓ Nenhum valor nulo encontrado.")

# === 3. Análise de Status dos Pedidos ===

print("\n[3/9] Análise de Status dos Pedidos...")

status_counts = (
    master
    .drop_duplicates('order_id')
    .groupby('order_status')['order_id']
    .count()
    .sort_values(ascending=False)
)
total_orders = status_counts.sum()
print(f"\n  Distribuição de Status ({total_orders:,} pedidos únicos):")
for status, count in status_counts.items():
    pct = count / total_orders * 100
    print(f"    {status:<30} {count:>7,}  ({pct:.1f}%)")

# === 4. Análise Temporal ===

print("\n[4/9] Análise Temporal...")

master_orders = master.drop_duplicates(subset='order_id').copy()
master_orders['year_month'] = (
    master_orders['order_purchase_timestamp']
    .dt.to_period('M')
    .astype(str)   # converter para str para serialização correta no CSV
)

orders_by_month = (
    master_orders
    .groupby('year_month')['order_id']
    .count()
    .sort_index()
)
print(f"\n  Pedidos por mês ({len(orders_by_month)} meses):")
print(orders_by_month.to_string())

print(f"\n  Período dos dados: "
      f"{master_orders['order_purchase_timestamp'].min().date()} "
      f"até {master_orders['order_purchase_timestamp'].max().date()}")


# === 5. Análise de Preços e Valores ===

print("\n[5/9] Análise de Preços e Valores...")

for col, label in [
    ('price',               'Preço (R$)'),
    ('freight_value',       'Frete (R$)'),
    ('payment_value_total', 'Pagamento Total (R$)')
]:
    if col in master.columns:
        s = master[col].dropna()
        print(f"\n  {label}:")
        print(f"    Média:    R$ {s.mean():>10.2f}")
        print(f"    Mediana:  R$ {s.median():>10.2f}")
        print(f"    Mín:      R$ {s.min():>10.2f}")
        print(f"    Máx:      R$ {s.max():>10.2f}")
        print(f"    Desvio:   R$ {s.std():>10.2f}")

# === 6. Análise Geográfica ===

print("\n[6/9] Análise Geográfica...")

print("\n  Top 10 Estados de Clientes (pedidos únicos):")
top_customer_states = (
    master_orders
    .groupby('customer_state')['customer_id']
    .nunique()
    .sort_values(ascending=False)
    .head(10)
)
print(top_customer_states.to_string())

print("\n  Top 10 Cidades de Clientes (pedidos únicos):")
top_customer_cities = (
    master_orders
    .groupby('customer_city')['customer_id']
    .nunique()
    .sort_values(ascending=False)
    .head(10)
)
print(top_customer_cities.to_string())

if 'seller_state' in master.columns:
    print("\n  Top 10 Estados de Vendedores:")
    top_seller_states = (
        master
        .groupby('seller_state')['seller_id']
        .nunique()
        .sort_values(ascending=False)
        .head(10)
    )
    print(top_seller_states.to_string())

# === 7. Análise de Reviews ===

print("\n[7/9] Análise de Reviews...")

# Usar review_score (inteiro 1-5); fallback para review_score_mean
review_col = 'review_score' if 'review_score' in master.columns else 'review_score_mean'

if review_col in master.columns:
    review_dist = (
        master_orders[review_col]
        .dropna()
        .round()
        .astype(int)
        .value_counts()
        .sort_index()
    )
    total_reviews = review_dist.sum()
    print(f"\n  Distribuição de Review Scores ({total_reviews:,} pedidos com review):")
    for score, count in review_dist.items():
        pct = count / total_reviews * 100
        bar = '█' * int(pct / 2)
        print(f"    {score} ★  {count:>7,}  ({pct:.1f}%)  {bar}")

    print(f"\n  Score médio: {master_orders[review_col].mean():.2f}")
else:
    print("  ⚠ Coluna de review score não encontrada.")



# === 8. Análise de Atrasos na Entrega ===

print("\n[8/9] Análise de Atrasos na Entrega...")

delivered = (
    master_orders[
        (master_orders['order_status'] == 'delivered') &
        master_orders['order_delivered_customer_date'].notna() &
        master_orders['order_estimated_delivery_date'].notna()
    ]
    .copy()
)

delivered['delivery_delay_days'] = (
    delivered['order_delivered_customer_date'] -
    delivered['order_estimated_delivery_date']
).dt.total_seconds() / (24 * 3600)

delivered['is_delayed'] = delivered['delivery_delay_days'] > 0

n_delivered  = len(delivered)
n_delayed    = delivered['is_delayed'].sum()
pct_delayed  = n_delayed / n_delivered * 100 if n_delivered > 0 else 0

print(f"\n  Pedidos entregues analisados: {n_delivered:,}")
print(f"  Pedidos atrasados:            {n_delayed:,}  ({pct_delayed:.1f}%)")
print(f"  Pedidos no prazo:             {n_delivered - n_delayed:,}  ({100 - pct_delayed:.1f}%)")
print(f"\n  Atraso médio (dias):   {delivered['delivery_delay_days'].mean():.2f}")
print(f"  Atraso mediano (dias): {delivered['delivery_delay_days'].median():.2f}")
print(f"  Atraso máximo (dias):  {delivered['delivery_delay_days'].max():.2f}")
print(f"  Adiantamento máx (dias): {delivered['delivery_delay_days'].min():.2f}")

# === 9. Análise de Categorias e Pagamentos ===

print("\n[9/9] Análise de Categorias e Métodos de Pagamento...")

print("\n  Top 15 Categorias Mais Vendidas:")
if 'product_category_name_english' in master.columns:
    top_categories = master['product_category_name_english'].value_counts().head(15)
    print(top_categories.to_string())
else:
    print("  ⚠ Coluna 'product_category_name_english' não encontrada.")

print("\n  Distribuição de Métodos de Pagamento (pedidos únicos):")
if 'payment_type_main' in master_orders.columns:
    payment_dist = (
        master_orders
        .groupby('payment_type_main')['order_id']
        .count()
        .sort_values(ascending=False)
    )
    total_pay = payment_dist.sum()
    for method, count in payment_dist.items():
        pct = count / total_pay * 100
        print(f"    {method:<25} {count:>7,}  ({pct:.1f}%)")
else:
    print("  ⚠ Coluna 'payment_type_main' não encontrada.")

# === 10. Salvar Datasets Processados ===

print(f"\nSalvando datasets processados em: {DATA_DIR}")

# Salvar apenas pedidos únicos (sem duplicatas de itens)
delivered.to_csv(DATA_DIR / 'delivered_orders_with_delay.csv', index=False)
print("  ✓ delivered_orders_with_delay.csv")

master_orders.to_csv(DATA_DIR / 'master_orders_unique.csv', index=False)
print("  ✓ master_orders_unique.csv")

print("\n" + "=" * 80)
print("  FASE 2 - ANÁLISE EXPLORATÓRIA CONCLUÍDA COM SUCESSO!")
print("=" * 80)