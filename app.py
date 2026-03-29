"""
Análise Preditiva de E-Commerce
Integra todas as 6 etapas de ML

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
from typing import Optional
import re
import warnings

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Brazilian E-Commerce Public Dataset by Olist",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caminhos
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Utilitários com cache
@st.cache_data(show_spinner=False)
def load_csv_cached(path_str: str, **kwargs) -> Optional[pd.DataFrame]:
    """Carrega CSV com cache. Retorna None se o arquivo não existir."""
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path, **kwargs)


@st.cache_resource(show_spinner=False)
def load_model_cached(path_str: str):
    """Carrega artefato joblib com cache. Retorna None se não existir."""
    path = Path(path_str)
    if not path.exists():
        return None
    return joblib.load(path)


def load_metrics(path: Path) -> Optional[pd.DataFrame]:
    """Carrega CSV de métricas com mensagem de erro amigável."""
    df = load_csv_cached(str(path))
    if df is None:
        st.warning(f"⚠️ Arquivo de métricas não encontrado: `{path.name}`. Execute o script de treino correspondente.")
    return df


def read_html_file(path: Path) -> Optional[str]:
    """Lê arquivo HTML. Retorna None se não existir."""
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# SIDEBAR — NAVEGAÇÃO
st.sidebar.title("🛒 Pinheiro DataWorks - ML Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegação",
    [
        "🏠 Home - Overview",
        "📦 Etapa 01 - Previsão de Atraso",
        "👥 Etapa 02 - Churn & LTV",
        "💬 Etapa 03 - Análise de Sentimento",
        "🎁 Etapa 04 - Recomendação",
        "💰 Etapa 05 - Precificação Inteligente",
        "🏪 Etapa 06 - Clustering de Sellers"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Tecnologias:**  
- Python, XGBoost, Scikit-learn  
- Streamlit, Plotly  
- Machine Learning & NLP
""")


# PÁGINA HOME — OVERVIEW
if page == "🏠 Home - Overview":
    st.markdown(
        '<div class="main-header">📊 Brazilian E-Commerce Public Dataset by Olist</div>',
        unsafe_allow_html=True
    )
    st.markdown("### Plataforma Completa de Análise Preditiva para E-Commerce")

    # KPIs
    kpis = load_csv_cached(str(BASE_DIR / 'kpis.csv'))
    if kpis is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📦 Total de Pedidos",  f"{int(kpis['total_orders'].iloc[0]):,}")
        with col2:
            st.metric("💰 Receita Total",      f"R$ {kpis['total_revenue'].iloc[0]:,.2f}")
        with col3:
            st.metric("👥 Clientes Únicos",    f"{int(kpis['total_customers'].iloc[0]):,}")
        with col4:
            st.metric("🏪 Vendedores",         f"{int(kpis['total_sellers'].iloc[0]):,}")
        with col5:
            st.metric("🎫 Ticket Médio",       f"R$ {kpis['avg_ticket'].iloc[0]:.2f}")
    else:
        st.warning("⚠️ `kpis.csv` não encontrado. Execute `create_visualizations.py` primeiro.")

    st.markdown("---")

    # Gráficos
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Evolução de Vendas ao Longo do Tempo")
        html_sales = read_html_file(BASE_DIR / 'sales_over_time.html')
        if html_sales:
            st.components.v1.html(html_sales, height=450)
        else:
            st.warning("⚠️ `sales_over_time.html` não encontrado.")

    with col2:
        st.subheader("🗺️ Concentração Geográfica de Clientes")
        master = load_csv_cached(str(DATA_DIR / 'master_table_v2.csv'))
        if master is not None and 'customer_state' in master.columns:
            id_col = 'customer_unique_id' if 'customer_unique_id' in master.columns else 'customer_id'
            state_counts = (
                master.groupby('customer_state')[id_col]
                .nunique()
                .reset_index(name='clientes')
                .sort_values('clientes', ascending=False)
                .head(10)
            )
            fig = px.bar(
                state_counts, x='customer_state', y='clientes',
                labels={'customer_state': 'Estado', 'clientes': 'Clientes'},
                color='clientes', color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ `master_table_v2.csv` não encontrado ou sem coluna `customer_state`.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Top 15 Categorias Mais Vendidas")
        html_cat = read_html_file(BASE_DIR / 'top_categories.html')
        if html_cat:
            st.components.v1.html(html_cat, height=450)
        else:
            st.warning("⚠️ `top_categories.html` não encontrado.")

    with col2:
        st.subheader("⭐ Distribuição de Review Scores")
        html_rev = read_html_file(BASE_DIR / 'review_distribution.html')
        if html_rev:
            st.components.v1.html(html_rev, height=450)
        else:
            st.warning("⚠️ `review_distribution.html` não encontrado.")

    st.markdown("---")
    st.success("✅ Dashboard carregado! Use a navegação lateral para explorar os modelos de ML.")


# ETAPA 01 — PREVISÃO DE ATRASO
elif page == "📦 Etapa 01 - Previsão de Atraso":
    st.title("📦 Etapa 01: Previsão de Atraso na Entrega")
    st.markdown("### Modelo de Classificação Binária — XGBoost")

    metrics = load_metrics(MODEL_DIR / 'etapa01_metrics.csv')
    if metrics is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy",  f"{metrics['accuracy'].iloc[0]:.2%}")
        with col2:
            st.metric("Precision", f"{metrics['precision'].iloc[0]:.2%}")
        with col3:
            st.metric("Recall",    f"{metrics['recall'].iloc[0]:.2%}")
        with col4:
            st.metric("F1-Score",  f"{metrics['f1_score'].iloc[0]:.4f}")
        with col5:
            st.metric("AUC-ROC",   f"{metrics['auc_roc'].iloc[0]:.4f}")

    st.markdown("---")
    st.subheader("🔮 Simular Previsão de Atraso")

    col1, col2, col3 = st.columns(3)
    with col1:
        distance      = st.slider("Distância Cliente-Seller (km)", 0, 5000, 500)
        same_state    = st.selectbox("Mesmo Estado?", ["Sim", "Não"])
        num_items     = st.number_input("Número de Itens", 1, 10, 1)
    with col2:
        order_value   = st.number_input("Valor do Pedido (R$)", 10.0, 5000.0, 100.0)
        freight       = st.number_input("Frete (R$)", 0.0, 200.0, 15.0)
        promised_days = st.slider("Prazo Prometido (dias)", 1, 60, 15)
    with col3:
        product_weight = st.number_input("Peso do Produto (g)", 100, 50000, 1000)
        quarter        = st.selectbox("Trimestre", [1, 2, 3, 4])

    if st.button("🎯 Prever Probabilidade de Atraso", type="primary"):
        model_01    = load_model_cached(str(MODEL_DIR / 'etapa01_delay_model.pkl'))
        feat_cols_01 = load_model_cached(str(MODEL_DIR / 'etapa01_feature_cols.pkl'))

        if model_01 is not None and feat_cols_01 is not None:
            # Montar vetor de features com os MESMOS nomes usados no treino
            input_data = {
                'distance_customer_seller_km': distance,    # era distance_km
                'same_state':                  1 if same_state == "Sim" else 0,
                'order_item_id':               num_items,  # era order_items_qty
                'payment_value_total':         order_value, # era payment_value
                'freight_value':               freight,
                'promised_delivery_days':      promised_days, # era estimated_delivery_days
                'product_weight_g':            product_weight,
                'purchase_quarter':            quarter,
            }
            X_input = pd.DataFrame([{c: input_data.get(c, 0) for c in feat_cols_01}])
            prob_delay = float(model_01.predict_proba(X_input)[0][1])
        else:
            st.warning("⚠️ Modelo não encontrado. Exibindo simulação.")
            prob_delay = float(np.random.beta(2, 5))

        st.markdown("### 📊 Resultado da Previsão")
        if prob_delay > 0.5:
            st.error(f"⚠️ **ALTO RISCO DE ATRASO**: {prob_delay:.1%} de probabilidade")
            st.warning("**Recomendações:**")
            st.write("- Alertar o vendedor para priorizar este pedido")
            st.write("- Oferecer rastreamento premium ao cliente")
            st.write("- Considerar frete expresso")
        else:
            st.success(f"✅ **BAIXO RISCO DE ATRASO**: {prob_delay:.1%} de probabilidade")
            st.info("**Previsão:**")
            st.write("- Pedido tem alta chance de chegar no prazo")
            st.write("- Processo padrão de logística")

    st.markdown("---")
    st.info("**Modelo:** XGBoost Classifier com scale_pos_weight para balancear classes.")


# ETAPA 02 — CHURN & LTV
elif page == "👥 Etapa 02 - Churn & LTV":
    st.title("👥 Etapa 02: Previsão de Churn e LTV")
    st.markdown("### Modelos de Classificação (Churn) e Regressão (LTV)")

    metrics = load_metrics(MODEL_DIR / 'etapa02_metrics.csv')
    if metrics is not None:
        st.subheader("📊 Métricas dos Modelos")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🚨 Modelo de Churn")
            st.metric("AUC-ROC", f"{metrics['churn_auc_roc'].iloc[0]:.4f}")
            st.metric("F1-Score", f"{metrics['churn_f1'].iloc[0]:.4f}")
            recall_val = metrics['churn_recall'].iloc[0]
            # Normalizar para 0–1 caso venha como percentual
            if recall_val > 1:
                recall_val /= 100
            st.metric("Recall", f"{recall_val:.2%}")
        with col2:
            st.markdown("#### 💰 Modelo de LTV")
            st.metric("R²",   f"{metrics['ltv_r2'].iloc[0]:.4f}")
            st.metric("RMSE", f"R$ {metrics['ltv_rmse'].iloc[0]:.2f}")
            st.metric("MAE",  f"R$ {metrics['ltv_mae'].iloc[0]:.2f}")

    st.markdown("---")
    st.subheader("🔍 Análise de Cliente Individual")

    col1, col2, col3 = st.columns(3)
    with col1:
        num_orders        = st.number_input("Número de Pedidos", 1, 50, 3)
        num_products      = st.number_input("Produtos Únicos Comprados", 1, 50, 5)
    with col2:
        avg_order_value   = st.number_input("Ticket Médio por Pedido (R$)", 10.0, 5000.0, 100.0)
        avg_review        = st.slider("Review Médio", 1.0, 5.0, 4.0, 0.1)
    with col3:
        lifetime_days     = st.slider("Dias como Cliente", 1, 1000, 180)

    # Derivar frequência (pedidos/dia) a partir dos inputs
    frequency_per_day = num_orders / max(lifetime_days, 1)

    if st.button("📊 Analisar Cliente", type="primary"):
        model_churn      = load_model_cached(str(MODEL_DIR / 'etapa02_churn_model.pkl'))
        model_ltv        = load_model_cached(str(MODEL_DIR / 'etapa02_ltv_model.pkl'))
        feat_cols_churn  = load_model_cached(str(MODEL_DIR / 'etapa02_feature_cols.pkl'))
        feat_cols_ltv    = load_model_cached(str(MODEL_DIR / 'etapa02_ltv_feature_cols.pkl'))

        # Mapa de features alinhado com CHURN_FEATURE_COLS do treino
        input_churn = {
            'num_orders':              num_orders,
            'num_unique_products':     num_products,
            'frequency_orders_per_day':frequency_per_day,
            'avg_order_value':         avg_order_value,
            'avg_review_score':        avg_review,
            'customer_state':          0,   # desconhecido → encoding neutro
            'favorite_category':       0,   # desconhecido → encoding neutro
        }

        # Mapa de features alinhado com LTV_FEATURE_COLS do treino
        input_ltv = {
            'num_orders':              num_orders,
            'num_unique_products':     num_products,
            'frequency_orders_per_day':frequency_per_day,
            'avg_order_value':         avg_order_value,
            'avg_review_score':        avg_review,
            'customer_lifetime_days':  lifetime_days,
            'customer_state':          0,
            'favorite_category':       0,
        }

        if model_churn is not None and feat_cols_churn is not None:
            X_churn    = pd.DataFrame([{c: input_churn.get(c, 0) for c in feat_cols_churn}])
            churn_prob = float(model_churn.predict_proba(X_churn)[0][1])
        else:
            st.warning("⚠️ Modelo de churn não encontrado. Exibindo estimativa.")
            churn_prob = min((365 - lifetime_days) / 365, 0.95) if lifetime_days < 365 else 0.1

        if model_ltv is not None and feat_cols_ltv is not None:
            X_ltv_df = pd.DataFrame([{c: input_ltv.get(c, 0) for c in feat_cols_ltv}])
            ltv_pred  = float(model_ltv.predict(X_ltv_df)[0])
        else:
            ltv_pred = avg_order_value * num_orders * 1.25

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🚨 Status de Churn")
            if churn_prob > 0.5:
                st.error(f"⚠️ **CLIENTE CHURNADO**: {churn_prob:.1%} de probabilidade")
                st.warning("**Ações Recomendadas:**")
                st.write("- 🎁 Cupom de reativação de 20%")
                st.write("- 📧 Campanha de e-mail personalizada")
                st.write("- 📱 SMS com oferta especial")
            else:
                st.success(f"✅ **CLIENTE ATIVO**: {(1 - churn_prob):.1%} de retenção")
                st.info("**Estratégias:**")
                st.write("- 🌟 Programa de fidelidade")
                st.write("- 🎯 Cross-sell de produtos complementares")

        with col2:
            st.markdown("### 💰 Lifetime Value (LTV)")
            st.metric("LTV Predito",  f"R$ {ltv_pred:.2f}")
            st.metric("Ticket Médio", f"R$ {avg_order_value:.2f}")

            # Segmentação dinâmica por LTV
            if ltv_pred > 500:
                segment, color = "🌟 VIP",     "green"
            elif ltv_pred > 200:
                segment, color = "💎 Premium", "blue"
            else:
                segment, color = "🥉 Bronze",  "gray"
            st.markdown(f"**Segmento:** :{color}[{segment}]")


# ETAPA 03 — ANÁLISE DE SENTIMENTO
elif page == "💬 Etapa 03 - Análise de Sentimento":
    st.title("💬 Etapa 03: Análise de Sentimento em Reviews")
    st.markdown("### Modelo de NLP — TF-IDF + Logistic Regression")

    metrics = load_metrics(MODEL_DIR / 'etapa03_metrics.csv')
    if metrics is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy",          f"{metrics['accuracy'].iloc[0]:.2%}")
        with col2:
            st.metric("F1-Score Macro",    f"{metrics['f1_macro'].iloc[0]:.4f}")
        with col3:
            st.metric("F1-Score Weighted", f"{metrics['f1_weighted'].iloc[0]:.4f}")

    st.markdown("---")

    # Distribuição de sentimentos
    st.subheader("📊 Distribuição de Sentimentos")
    sentiment_data = load_csv_cached(str(DATA_DIR / 'etapa03_sentiment_dataset.csv'))
    if sentiment_data is not None and 'sentiment' in sentiment_data.columns:
        sentiment_counts = sentiment_data['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='Proporção de Sentimentos nas Reviews',
            color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Dataset de sentimento não encontrado.")

    st.markdown("---")
    st.subheader("🔮 Analisar Sentimento de Review")

    review_text = st.text_area(
        "Digite o texto da review:",
        placeholder="Ex: Produto excelente, chegou antes do prazo!",
        height=100
    )

    if st.button("🎯 Analisar Sentimento", type="primary"):
        if review_text.strip():
            model_sent  = load_model_cached(str(MODEL_DIR / 'etapa03_sentiment_model.pkl'))
            vectorizer  = load_model_cached(str(MODEL_DIR / 'etapa03_vectorizer.pkl'))
            model_cfg   = load_model_cached(str(MODEL_DIR / 'etapa03_model_config.pkl'))

            if model_sent is not None and vectorizer is not None:
                # Limpeza idêntica ao treino
                clean_text = re.sub(r'[^a-záàâãäéèêëíìîïóòôõöúùûüçñ\s]', ' ', review_text.lower())
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                X_sent     = vectorizer.transform([clean_text])
                pred_label = model_sent.predict(X_sent)[0]
                pred_proba = model_sent.predict_proba(X_sent).max()
            else:
                # Fallback por palavras-chave
                st.warning("⚠️ Modelo não encontrado. Usando análise por palavras-chave.")
                positive_words = ['excelente', 'ótimo', 'perfeito', 'adorei', 'parabéns', 'recomendo', 'rápido']
                negative_words = ['ruim', 'péssimo', 'não', 'atraso', 'defeito', 'problema']
                text_lower = review_text.lower()
                pos_count  = sum(1 for w in positive_words if w in text_lower)
                neg_count  = sum(1 for w in negative_words if w in text_lower)
                if pos_count > neg_count:
                    pred_label, pred_proba = 'positive', 0.85
                elif neg_count > pos_count:
                    pred_label, pred_proba = 'negative', 0.80
                else:
                    pred_label, pred_proba = 'neutral',  0.65

            label_map = {
                'positive': ("✅", "Positivo",  "green"),
                'negative': ("❌", "Negativo",  "red"),
                'neutral':  ("⚪", "Neutro",    "gray"),
            }
            icon, label_pt, color = label_map.get(pred_label, ("❓", pred_label, "gray"))
            st.markdown(f"### {icon} Sentimento: :{color}[{label_pt}] ({pred_proba:.1%} confiança)")

            if pred_label == 'negative':
                st.error("⚠️ **Alerta:** Review negativa detectada!")
                st.warning("**Ações Recomendadas:**")
                st.write("- 📞 Entrar em contato com o cliente em até 24h")
                st.write("- 🎁 Oferecer compensação/desconto")
                st.write("- 📊 Registrar feedback para melhoria")
        else:
            st.warning("Por favor, digite uma review para análise.")

    st.markdown("---")
    st.subheader("☁️ Palavras Mais Frequentes por Sentimento")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ✅ Palavras Positivas")
        st.success("parabéns, perfeito, adorei, excelente, rápido, antes do prazo, conforme, lindo")
    with col2:
        st.markdown("#### ❌ Palavras Negativas")
        st.error("não recomendo, péssima, atraso, defeito, problema, dinheiro, esperar, ruim")


# ETAPA 04 — RECOMENDAÇÃO
elif page == "🎁 Etapa 04 - Recomendação":
    st.title("🎁 Etapa 04: Sistema de Recomendação de Produtos")
    st.markdown("### Modelo Baseado em Popularidade e Interações")

    top_products = load_model_cached(str(MODEL_DIR / 'etapa04_top_products.pkl'))
    if top_products is None:
        st.warning("⚠️ `etapa04_top_products.pkl` não encontrado. Execute `train_etapa04_05_06.py`.")

    st.subheader("🔥 Top 20 Produtos Mais Populares")

    master = load_csv_cached(str(DATA_DIR / 'master_table_v2.csv'))
    if master is not None and 'product_id' in master.columns:
        product_sales = (
            master.groupby('product_id')
            .size()
            .reset_index(name='sales_count')
            .sort_values('sales_count', ascending=False)
            .head(20)
        )
        if 'product_category_name_english' in master.columns:
            product_sales = product_sales.merge(
                master[['product_id', 'product_category_name_english']].drop_duplicates(),
                on='product_id', how='left'
            )
        fig = px.bar(
            product_sales, x='sales_count', y='product_id',
            orientation='h',
            labels={'sales_count': 'Número de Vendas', 'product_id': 'Produto'},
            color='sales_count', color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ `master_table_v2.csv` não encontrado.")

    st.markdown("---")
    st.subheader("🎯 Recomendações Personalizadas")

    col1, col2 = st.columns(2)
    with col1:
        customer_history = st.multiselect(
            "Selecione categorias de produtos comprados anteriormente:",
            ['bed_bath_table', 'health_beauty', 'sports_leisure', 'computers_accessories',
             'furniture_decor', 'housewares', 'watches_gifts', 'toys', 'electronics'],
            default=['health_beauty']
        )

    with col2:
        st.markdown("#### 💡 Produtos Recomendados:")
        if customer_history:
            st.success("✅ Com base no seu histórico, recomendamos:")
            recommendations = {
                'health_beauty':         ['Perfumes', 'Cosméticos Premium', 'Suplementos'],
                'sports_leisure':        ['Equipamentos de Ginástica', 'Bicicletas', 'Acessórios Esportivos'],
                'computers_accessories': ['Mouses', 'Teclados', 'Webcams'],
                'bed_bath_table':        ['Jogos de Lençóis', 'Toalhas Premium', 'Almofadas'],
                'furniture_decor':       ['Quadros Decorativos', 'Vasos', 'Luminárias']
            }
            for category in customer_history[:2]:
                if category in recommendations:
                    for item in recommendations[category][:2]:
                        st.write(f"- 🎁 {item}")
        else:
            st.info("Selecione categorias para ver recomendações.")

    st.markdown("---")
    st.info("**Modelo:** Sistema híbrido baseado em popularidade e filtragem colaborativa.")


# ETAPA 05 — PRECIFICAÇÃO INTELIGENTE
elif page == "💰 Etapa 05 - Precificação Inteligente":
    st.title("💰 Etapa 05: Precificação Inteligente com Elasticidade")
    st.markdown("### Modelo de Regressão — XGBoost")

    metrics = load_metrics(MODEL_DIR / 'etapa05_metrics.csv')
    if metrics is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R²",   f"{metrics['r2'].iloc[0]:.4f}")
        with col2:
            st.metric("RMSE", f"R$ {metrics['rmse'].iloc[0]:.2f}")
        with col3:
            st.metric("MAE",  f"R$ {metrics['mae'].iloc[0]:.2f}")
        with col4:
            if 'mape' in metrics.columns:
                st.metric("MAPE", f"{metrics['mape'].iloc[0]:.2f}%")

    st.markdown("---")
    st.subheader("💡 Simulador de Preço Inteligente")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📦 Características do Produto")
        category = st.selectbox(
            "Categoria",
            ['health_beauty', 'sports_leisure', 'computers_accessories',
             'bed_bath_table', 'furniture_decor', 'housewares']
        )
        weight  = st.slider("Peso (g)",       100, 10000, 1000)
        volume  = st.slider("Volume (cm³)",   100, 50000, 5000)
        photos  = st.slider("Número de Fotos", 1,     10,    3)
    with col2:
        st.markdown("#### 🚚 Informações Adicionais")
        freight = st.number_input("Frete (R$)", 0.0, 100.0, 15.0)
        demand  = st.slider("Demanda do Produto (vendas/mês)", 1, 500, 50)

    if st.button("💰 Calcular Preço Sugerido", type="primary"):
        model_pricing   = load_model_cached(str(MODEL_DIR / 'etapa05_pricing_model.pkl'))
        feat_cols_05    = load_model_cached(str(MODEL_DIR / 'etapa05_feature_cols.pkl'))
        encoders_05     = load_model_cached(str(MODEL_DIR / 'etapa05_label_encoders.pkl'))

        if model_pricing is not None and feat_cols_05 is not None:
            input_pricing = {
                'product_weight_g':              weight,
                'product_volume_cm3':            volume,
                'product_photos_qty':            photos,
                'freight_value':                 freight,
                'product_demand_count':          demand,
                'product_category_name_english': category,
            }
            # Aplicar encoding salvo
            if encoders_05:
                for col, mapping in encoders_05.items():
                    if col in input_pricing:
                        input_pricing[col] = mapping.get(str(input_pricing[col]), -1)
            X_price   = pd.DataFrame([{c: input_pricing.get(c, 0) for c in feat_cols_05}])
            base_price = float(model_pricing.predict(X_price)[0])
        else:
            st.warning("⚠️ Modelo não encontrado. Exibindo estimativa.")
            base_price = (weight * 0.05 + volume * 0.001 + freight * 2 + photos * 5) * (1 + demand / 1000)

        st.markdown("### 📊 Análise de Precificação")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💵 Preço Sugerido",         f"R$ {base_price:.2f}")
        with col2:
            st.metric("🔥 Preço Competitivo (-10%)", f"R$ {base_price * 0.9:.2f}")
        with col3:
            st.metric("⭐ Preço Premium (+15%)",     f"R$ {base_price * 1.15:.2f}")

        st.markdown("---")
        st.subheader("📈 Simulação de Elasticidade Preço-Demanda")

        price_range   = np.linspace(base_price * 0.7, base_price * 1.3, 20)
        demand_range  = demand * (1 - (price_range - base_price) / base_price * 0.8)
        revenue_range = price_range * demand_range

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=price_range, y=demand_range,  name='Demanda',      mode='lines'), secondary_y=False)
        fig.add_trace(go.Scatter(x=price_range, y=revenue_range, name='Receita (R$)', mode='lines'), secondary_y=True)
        fig.update_xaxes(title_text='Preço (R$)')
        fig.update_yaxes(title_text='Demanda (unidades)', secondary_y=False)
        fig.update_yaxes(title_text='Receita Total (R$)', secondary_y=True)
        fig.update_layout(title='Curva de Elasticidade: Preço vs Demanda vs Receita', height=400)
        st.plotly_chart(fig, use_container_width=True)

        optimal_idx     = int(np.argmax(revenue_range))
        optimal_price   = price_range[optimal_idx]
        optimal_revenue = revenue_range[optimal_idx]
        st.success(f"🎯 **Preço Ótimo para Maximizar Receita:** R$ {optimal_price:.2f}  (Receita: R$ {optimal_revenue:.2f})")


# ETAPA 06 — CLUSTERING DE SELLERS
elif page == "🏪 Etapa 06 - Clustering de Sellers":
    st.title("🏪 Etapa 06: Clustering e Análise de Sellers")
    st.markdown("### Modelo de Clustering — K-Means")

    metrics = load_metrics(MODEL_DIR / 'etapa06_metrics.csv')
    # Sellers com clusters agora em DATA_DIR (corrigido no train_etapa04_05_06.py)
    sellers_clustered = load_csv_cached(str(DATA_DIR / 'etapa06_sellers_with_clusters.csv'))

    if metrics is not None and sellers_clustered is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Número de Clusters", int(metrics['best_k'].iloc[0]))
        with col2:
            st.metric("Silhouette Score",   f"{metrics['silhouette_score'].iloc[0]:.4f}")
        with col3:
            st.metric("Total de Sellers",   f"{len(sellers_clustered):,}")
    else:
        st.warning("⚠️ Dados de clustering não encontrados. Execute `train_etapa04_05_06.py`.")

    if sellers_clustered is not None and 'cluster' in sellers_clustered.columns:
        st.markdown("---")
        st.subheader("📊 Distribuição de Sellers por Cluster")

        cluster_dist = sellers_clustered['cluster'].value_counts().sort_index()
        fig = px.bar(
            x=cluster_dist.index, y=cluster_dist.values,
            labels={'x': 'Cluster', 'y': 'Número de Sellers'},
            color=cluster_dist.values, color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🎯 Perfil dos Clusters")

        agg_cols = [c for c in ['num_orders', 'total_revenue', 'avg_review_score', 'delay_rate']
                    if c in sellers_clustered.columns]
        cluster_summary = sellers_clustered.groupby('cluster')[agg_cols].mean().round(2)

        # Limiar dinâmico para classificação (percentil 66 da receita média)
        if 'total_revenue' in cluster_summary.columns:
            revenue_p66 = cluster_summary['total_revenue'].quantile(0.66)
            revenue_p33 = cluster_summary['total_revenue'].quantile(0.33)
        else:
            revenue_p66, revenue_p33 = 10000, 3000

        for cluster_id in cluster_summary.index:
            n_sellers = cluster_dist.get(cluster_id, 0)
            with st.expander(f"📌 Cluster {cluster_id} — {n_sellers:,} Sellers"):
                cols = st.columns(len(agg_cols))
                labels_map = {
                    'num_orders':       ("Pedidos Médios",  "{:.0f}"),
                    'total_revenue':    ("Receita Média",   "R$ {:.2f}"),
                    'avg_review_score': ("Review Médio",    "{:.2f} ⭐"),
                    'delay_rate':       ("Taxa de Atraso",  "{:.1%}"),
                }
                for i, col_name in enumerate(agg_cols):
                    label, fmt = labels_map.get(col_name, (col_name, "{}"))
                    val = cluster_summary.loc[cluster_id, col_name]
                    cols[i].metric(label, fmt.format(val))

                # Classificação dinâmica por percentil de receita
                if 'total_revenue' in cluster_summary.columns:
                    rev = cluster_summary.loc[cluster_id, 'total_revenue']
                    if rev >= revenue_p66:
                        st.success("🌟 **Sellers Premium** — Alto volume, alta receita")
                        st.write("**Ações:** Manter engajamento, oferecer benefícios exclusivos")
                    elif rev >= revenue_p33:
                        st.info("💎 **Sellers em Crescimento** — Volume intermediário")
                        st.write("**Ações:** Incentivos de expansão, suporte personalizado")
                    else:
                        st.warning("🥉 **Sellers Iniciantes** — Baixo volume")
                        st.write("**Ações:** Treinamento, suporte, incentivos para crescimento")

        st.markdown("---")
        st.subheader("🔍 Análise Individual de Seller")

        seller_id_input = st.text_input("Digite o ID do Seller:", placeholder="Ex: abc123...")

        if st.button("📊 Analisar Seller", type="primary"):
            if seller_id_input.strip():
                seller_col = 'seller_id' if 'seller_id' in sellers_clustered.columns else None
                if seller_col and seller_id_input in sellers_clustered[seller_col].values:
                    row = sellers_clustered[sellers_clustered[seller_col] == seller_id_input].iloc[0]
                    cluster_id = int(row['cluster'])
                    st.success(f"✅ Seller encontrado — Cluster {cluster_id}")
                    cols = st.columns(len(agg_cols))
                    for i, col_name in enumerate(agg_cols):
                        label, fmt = labels_map.get(col_name, (col_name, "{}"))
                        cols[i].metric(label, fmt.format(row[col_name]))
                else:
                    st.info("**Demonstração:** Seller não encontrado no dataset. Exibindo análise fictícia.")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Cluster Atribuído", "Cluster 0")
                        st.metric("Total de Vendas",   "45")
                    with col2:
                        st.metric("Receita Total",     "R$ 5.420,00")
                        st.metric("Review Médio",      "4.2 ⭐")
                    with col3:
                        st.metric("Taxa de Atraso",    "6.5%")
                        st.metric("Estados Atendidos", "8")
                    st.warning("**Recomendações:**")
                    st.write("- 📈 Expandir catálogo de produtos")
                    st.write("- 🚀 Melhorar tempo de entrega")
                    st.write("- 🎯 Focar em marketing digital")
            else:
                st.warning("Digite um ID de seller para análise.")

st.markdown("---")
st.markdown("**© 2026 E-Commerce ML Analytics Dashboard | Desenvolvido por Renan Pinheiro**")