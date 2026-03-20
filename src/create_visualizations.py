"""
Criar visualizações para o Dashboard

"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import requests
import warnings

warnings.filterwarnings('ignore')

# Caminhos
# A pasta "data" e os arquivos de saída ficam na raiz do projeto.
BASE_DIR   = Path(__file__).resolve().parent.parent  # raiz do projeto
DATA_DIR   = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR

print("=" * 55)
print("  Criando visualizações para o dashboard...")
print("=" * 55)

# Carregar dados
master = pd.read_csv(
    DATA_DIR / 'master_table_v2.csv',
    parse_dates=['order_purchase_timestamp']
)
customers_geo = pd.read_csv(DATA_DIR / 'customers_geo.csv')


# 1. Mapa do Brasil — Concentração de Clientes por Estado
print("\n[1/5] Criando mapa de concentração geográfica...")

state_counts = (
    master
    .groupby('customer_state')['customer_id']
    .nunique()
    .reset_index(name='num_customers')
    .sort_values('num_customers', ascending=False)
)

BRAZIL_GEOJSON_URL = (
    "https://raw.githubusercontent.com/codeforamerica/"
    "click_that_hood/master/public/data/brazil-states.geojson"
)

try:
    brazil_geo = requests.get(BRAZIL_GEOJSON_URL, timeout=10).json()
except requests.RequestException as e:
    raise RuntimeError(f"Falha ao baixar GeoJSON do Brasil: {e}")

fig_map = px.choropleth(
    state_counts,
    geojson=brazil_geo,
    locations='customer_state',
    featureidkey='properties.sigla',
    color='num_customers',
    title='Concentração de Clientes por Estado no Brasil',
    color_continuous_scale='Blues',
    labels={'num_customers': 'Número de Clientes', 'customer_state': 'Estado'}
)
fig_map.update_geos(fitbounds="locations", visible=False)
fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

fig_map.write_html(OUTPUT_DIR / 'map_customers_by_state.html', include_plotlyjs='cdn')
print(f"  ✓ Mapa salvo: map_customers_by_state.html")

# 2. KPIs Principais
print("\n[2/5] Calculando KPIs principais...")

# Um registro por pedido para evitar duplicidade de pagamento
master_orders = master.drop_duplicates('order_id')

total_orders    = master_orders['order_id'].nunique()
total_revenue   = master_orders['payment_value_total'].sum()
total_customers = master_orders['customer_id'].nunique()
total_sellers   = master['seller_id'].nunique()          # usa master completo
avg_ticket      = total_revenue / total_orders if total_orders > 0 else 0

kpis = {
    'total_orders':    total_orders,
    'total_revenue':   round(total_revenue, 2),
    'total_customers': total_customers,
    'total_sellers':   total_sellers,
    'avg_ticket':      round(avg_ticket, 2)
}

pd.DataFrame([kpis]).to_csv(OUTPUT_DIR / 'kpis.csv', index=False)
print(f"  ✓ KPIs salvos: kpis.csv")

# 3. Vendas por Mês (Pedidos e Receita)
print("\n[3/5] Criando gráfico de vendas temporais...")

master_orders = master_orders.copy()
master_orders['year_month'] = (
    master_orders['order_purchase_timestamp']
    .dt.to_period('M')
    .astype(str)
)

sales_by_month = (
    master_orders
    .groupby('year_month')
    .agg(
        num_orders=('order_id', 'nunique'),
        revenue=('payment_value_total', 'sum')
    )
    .reset_index()
    .sort_values('year_month')
)

fig_sales = make_subplots(specs=[[{"secondary_y": True}]])

fig_sales.add_trace(
    go.Scatter(
        x=sales_by_month['year_month'],
        y=sales_by_month['num_orders'],
        name='Pedidos',
        mode='lines+markers',
        line=dict(color='steelblue')
    ),
    secondary_y=False
)

fig_sales.add_trace(
    go.Scatter(
        x=sales_by_month['year_month'],
        y=sales_by_month['revenue'],
        name='Receita (R$)',
        mode='lines+markers',
        line=dict(color='darkorange')
    ),
    secondary_y=True
)

fig_sales.update_xaxes(title_text='Mês', tickangle=45)
fig_sales.update_yaxes(title_text='Número de Pedidos', secondary_y=False)
fig_sales.update_yaxes(title_text='Receita (R$)', secondary_y=True)
fig_sales.update_layout(
    title='Evolução de Pedidos e Receita ao Longo do Tempo',
    height=450,
    hovermode='x unified'
)

fig_sales.write_html(OUTPUT_DIR / 'sales_over_time.html', include_plotlyjs='cdn')
print(f"  ✓ Gráfico de vendas salvo: sales_over_time.html")

# 4. Top Categorias
print("\n[4/5] Criando gráfico de categorias...")

top_categories = (
    master['product_category_name_english']
    .value_counts()
    .head(15)
    .reset_index()
)
top_categories.columns = ['category', 'count']

fig_categories = px.bar(
    top_categories,
    x='count',
    y='category',
    orientation='h',
    title='Top 15 Categorias Mais Vendidas',
    labels={'count': 'Número de Vendas', 'category': 'Categoria'},
    color='count',
    color_continuous_scale='Blues'
)
fig_categories.update_layout(
    yaxis={'categoryorder': 'total ascending'},
    coloraxis_showscale=False,
    height=500
)

fig_categories.write_html(OUTPUT_DIR / 'top_categories.html', include_plotlyjs='cdn')
print(f"  ✓ Gráfico de categorias salvo: top_categories.html")

# 5. Distribuição de Reviews
print("\n[5/5] Criando gráfico de reviews...")

reviews = pd.read_csv(DATA_DIR / 'reviews_processed.csv')
review_dist = (
    reviews['review_score']
    .value_counts()
    .sort_index()
    .reset_index()
)
review_dist.columns = ['review_score', 'count']

fig_reviews = px.bar(
    review_dist,
    x='review_score',
    y='count',
    title='Distribuição de Review Scores',
    labels={'review_score': 'Review Score', 'count': 'Quantidade'},
    color='review_score',
    color_continuous_scale='RdYlGn',
    text='count'
)
fig_reviews.update_traces(textposition='outside')
fig_reviews.update_layout(
    coloraxis_showscale=False,
    xaxis=dict(tickmode='linear', dtick=1),
    height=400
)

fig_reviews.write_html(OUTPUT_DIR / 'review_distribution.html', include_plotlyjs='cdn')
print(f"  ✓ Gráfico de reviews salvo: review_distribution.html")

print("\n" + "=" * 55)
print("  ✓ Todas as visualizações criadas com sucesso!")
print("=" * 55)