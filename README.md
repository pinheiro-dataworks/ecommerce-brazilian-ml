# 🛒 Projeto de Análise Preditiva de E-Commerce com Machine Learning

## 📋 1- Visão Geral

Projeto completo de ciência de dados aplicado a e-commerce brasileiro (dados Olist), implementando **6 modelos de Machine Learning** para resolver problemas reais de negócio, com dashboard interativo em Streamlit.

## 🎯 2 -Objetivos do Projeto

1. **Previsão de Atraso na Entrega** - Classificação binária para identificar pedidos em risco
2. **Previsão de Churn e LTV** - Identificar clientes em risco e prever valor vitalício
3. **Análise de Sentimento** - NLP para classificar reviews em positivo/neutro/negativo
4. **Sistema de Recomendação** - Motor de recomendação personalizado
5. **Precificação Inteligente** - Modelo de precificação com simulação de elasticidade
6. **Clustering de Sellers** - Segmentação de vendedores por performance

## 📈 3 - Métricas dos Modelos

| Etapa | Modelo | Métrica Principal | Valor |
|-------|--------|-------------------|-------|
| 01 - Atraso | XGBoost Classifier | AUC-ROC | **0.7993** |
| 02 - Churn | XGBoost Classifier | AUC-ROC | **~0.70–0.80** ¹ |
| 02 - LTV | XGBoost Regressor | R² | **0.2259** |
| 03 - Sentimento | LogisticRegression + TF-IDF | Accuracy | **78.48%** |
| 04 - Recomendação | Sistema por Popularidade | - | Top-50 produtos |
| 05 - Precificação | XGBoost Regressor | R² | **0.5310** |
| 06 - Clustering | K-Means | Silhouette Score | **0.7505** |

> ¹ O valor original de AUC-ROC 1.0000 para Churn foi resultado de **data leakage** corrigido:
> `recency_days` era simultaneamente a definição do label e uma feature do modelo.
> Após a correção (split temporal + remoção de `recency_days` das features),
> o AUC realista situa-se entre 0.70–0.80 dependendo do período de corte.

## 🎨 4 - Funcionalidades do Dashboard

### 🏠 4.1 - Home - Overview
- KPIs principais (pedidos, receita, clientes, sellers)
- Evolução temporal de vendas
- Mapa de concentração geográfica
- Top categorias e distribuição de reviews

### 📦 4.2 - Etapa 01 - Previsão de Atraso
- Interface de simulação de previsão
- Análise de risco de atraso
- Recomendações automatizadas

### 👥 4.3 - Etapa 02 - Churn & LTV
- Score de propensão ao churn
- Previsão de Lifetime Value
- Segmentação de clientes (VIP, Premium, Bronze)
- Ações recomendadas por segmento

### 💬 4.4 - Etapa 03 - Análise de Sentimento
- Análise de sentimento em tempo real
- Distribuição de sentimentos
- Nuvem de palavras
- Alertas para reviews negativas

### 🎁 4.5 - Etapa 04 - Recomendação
- Top produtos mais populares
- Recomendações personalizadas por categoria
- Sistema de filtragem colaborativa

### 💰 4.6 - Etapa 05 - Precificação Inteligente
- Simulador de preço otimizado
- Análise de elasticidade preço-demanda
- Sugestão de preços competitivos e premium
- Maximização de receita

### 🏪 4.7 - Etapa 06 - Clustering de Sellers
- Segmentação de vendedores (2 clusters)
- Perfil detalhado por cluster
- Análise individual de seller
- Recomendações de crescimento

## 🛠️ 5 - Tecnologias Utilizadas

- **Python 3.9+**
- **Pandas, NumPy** - Manipulação de dados
- **Scikit-learn** - Machine Learning clássico
- **XGBoost, LightGBM** - Gradient Boosting
- **Transformers, PyTorch** - NLP (opcional)
- **Plotly** - Visualizações interativas
- **Streamlit** - Dashboard web
- **Joblib** - Serialização de modelos

## 📊 6 - Insights de Negócio

### 6.1 - Previsão de Atraso
- **Taxa de atraso**: 8.11% dos pedidos
- **Features chave**: Distância cliente-seller, trimestre, mesmo estado
- **Impacto**: Redução de reviews negativas, melhor experiência do cliente

### 6.2 - Churn e LTV
- **Taxa de churn**: 84.48% (>120 dias sem comprar)
- **LTV médio**: R$ 213,02 (corrigido: calculado sem duplicação de itens por pedido)
- **Segmentação**: Clientes VIP (LTV > R$ 500) representam oportunidade de retenção
- **Nota técnica**: Data leakage corrigido — `recency_days` removida das features de churn

### 6.3 - Análise de Sentimento
- **65% reviews positivas**, 27% negativas, 8% neutras
- **Palavras-chave negativas**: "não recomendo", "péssima", "atraso"
- **Oportunidade**: Sistema de alerta precoce para problemas

### 6.4 - Precificação
- **Preço médio**: R$ 120,65
- **Elasticidade**: Modelo permite simular impacto de preço na demanda
- **Otimização**: Maximização de receita através de precificação dinâmica

### 6.5 - Clustering de Sellers
- **Cluster 0** (98%): Sellers em crescimento (baixo volume)
- **Cluster 1** (2%): Sellers premium (alto volume, alta receita)
- **Oportunidade**: Programas de desenvolvimento para Cluster 0

## 🎯 7 - Próximos Passos

**Integração com API REST** para deploy em produção
**Implementação de BERT PT-BR** para melhor análise de sentimento
**Sistema de alertas automáticos** via e-mail/SMS
**Dashboard executivo** com KPIs estratégicos
**Modelo de forecasting** de demanda por categoria
**Análise de coorte** de clientes

## 📝 8 - Definições de Negócio

**Churn**: Cliente sem compras há mais de **120 dias**
**LTV (Lifetime Value)**: Soma total gasta pelo cliente no período
**Atraso**: Entrega após a data estimada prometida
**Review Positiva**: Score 4-5 estrelas
**Review Negativa**: Score 1-2 estrelas

## 📁 9 - Estrutura do Projeto

PROJECT_01_ECOMMERCE_BR/
├── src/                   # Scripts de treino e processamento
├── data/                  # Dados processados (não versionados)
├── app.py                 # Dashboard Streamlit
├── requirements.txt       # Dependências
└── README.md


## 👨‍💻 10 - Autor
**Renan Pinheiro - Cientista de Dados**  
Projeto desenvolvido com dados públicos do Olist (Brazilian E-Commerce)