"""
ETAPA 03 - Treinamento do Modelo de Análise de Sentimento

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# Caminhos portáteis
# ====
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
print("  ETAPA 03 - TREINAMENTO: ANÁLISE DE SENTIMENTO")
print("=" * 80)

# 1. Carregar dataset
print("\n[1/7] Carregando dataset...")

dataset_path = DATA_DIR / 'etapa03_sentiment_dataset.csv'
if not dataset_path.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {dataset_path}\n"
        "Execute feature_engineering.py antes deste script."
    )

df = pd.read_csv(dataset_path)

for col in ['sentiment', 'review_comment_message']:
    if col not in df.columns:
        raise ValueError(
            f"Coluna '{col}' não encontrada no dataset.\n"
            "Verifique se o feature_engineering.py foi executado corretamente."
        )

print(f"  ✓ Dataset carregado: {df.shape}")
print(f"\n  Distribuição de sentimentos (total):")
dist = df['sentiment'].value_counts()
for label, count in dist.items():
    print(f"    {label:<10} {count:>7,}  ({count / len(df) * 100:.1f}%)")


# 2. Preparar dados
print("\n[2/7] Preparando dados...")

# Remover reviews sem texto
df = df[df['review_comment_message'].notna()].copy()
df['review_comment_message'] = df['review_comment_message'].astype(str)

# Concatenar título + mensagem quando disponível (mais contexto para NLP)
if 'review_comment_title' in df.columns:
    df['review_comment_title'] = df['review_comment_title'].fillna('').astype(str)
    df['review_text'] = (df['review_comment_title'] + ' ' + df['review_comment_message']).str.strip()
    print("  ✓ Título + mensagem concatenados para NLP.")
else:
    df['review_text'] = df['review_comment_message']

# Limpeza de texto — regex completa para PT-BR
print("  Limpando textos...")
df['review_clean'] = (
    df['review_text']
    .str.lower()
    .str.replace(
        r'[^a-záàâãäéèêëíìîïóòôõöúùûüçñ\s]',
        ' ',
        regex=True
    )
    .str.replace(r'\s+', ' ', regex=True)  # colapsar espaços múltiplos
    .str.strip()
)

# Remover reviews que ficaram vazios após limpeza
df = df[df['review_clean'].str.len() > 0].copy()

X = df['review_clean'].values
y = df['sentiment'].values

print(f"  ✓ Textos limpos: {len(X):,} reviews")


# 3. Split estratificado
print("\n[3/7] Dividindo dataset (split estratificado)...")

# Split aleatório estratificado é adequado para NLP (sem dependência temporal forte)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Treino: {len(X_train):,}  |  Teste: {len(X_test):,}")

# Distribuição por split
train_dist = pd.Series(y_train).value_counts()
test_dist  = pd.Series(y_test).value_counts()
print(f"\n  Distribuição treino:")
for label, count in train_dist.items():
    print(f"    {label:<10} {count:>7,}  ({count / len(y_train) * 100:.1f}%)")
print(f"\n  Distribuição teste:")
for label, count in test_dist.items():
    print(f"    {label:<10} {count:>7,}  ({count / len(y_test) * 100:.1f}%)")


# 4. Vetorização TF-IDF
print("\n[4/7] Vetorizando textos com TF-IDF...")
print("  NOTA: Usando TF-IDF + LogisticRegression como baseline.")
print("        Para BERT PT-BR, substitua por pipeline HuggingFace Transformers.")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    strip_accents='unicode',
    sublinear_tf=True   # log(1 + tf) — melhora performance em textos longos
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print(f"  ✓ Vetorização concluída: {X_train_vec.shape}")


# 5. Treinar modelo
print("\n[5/7] Treinando modelo Logistic Regression...")

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    solver='lbfgs',      # eficiente para multiclasse
    # multi_class removido: depreciado no sklearn 1.5+ (comportamento padrão é multinomial)
)

model.fit(X_train_vec, y_train)
print("  ✓ Modelo treinado!")

# 6. Avaliar modelo
print("\n[6/7] Avaliando modelo...")

y_pred = model.predict(X_test_vec)

accuracy    = accuracy_score(y_test, y_pred)
f1_macro    = f1_score(y_test, y_pred, average='macro',    zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n" + "=" * 80)
print("  MÉTRICAS DE AVALIAÇÃO — ETAPA 03")
print("=" * 80)
print(f"  Accuracy:          {accuracy:.4f}")
print(f"  F1-Score Macro:    {f1_macro:.4f}")
print(f"  F1-Score Weighted: {f1_weighted:.4f}")

print("\n  Confusion Matrix:")
cm     = confusion_matrix(y_test, y_pred, labels=model.classes_)
labels = model.classes_
header = f"{'':>12}" + "".join(f"{l:>12}" for l in labels)
print(f"  {header}")
for i, row_label in enumerate(labels):
    row = "".join(f"{cm[i, j]:>12,}" for j in range(len(labels)))
    print(f"  {row_label:>12}{row}")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, labels=model.classes_, zero_division=0))

# Top palavras por sentimento
print("\n  Top 10 Palavras Mais Importantes por Sentimento:")
feature_names = vectorizer.get_feature_names_out()
for i, sentiment_label in enumerate(model.classes_):
    coef       = model.coef_[i]
    top_idx    = np.argsort(coef)[-10:][::-1]
    top_words  = [feature_names[idx] for idx in top_idx]
    print(f"\n    {sentiment_label.upper()}: {', '.join(top_words)}")


# 7. Salvar modelo e artefatos
print("\n[7/7] Salvando modelo e artefatos...")

joblib.dump(model,      MODEL_DIR / 'etapa03_sentiment_model.pkl')
joblib.dump(vectorizer, MODEL_DIR / 'etapa03_vectorizer.pkl')

# Salvar classes e configurações para inferência
model_config = {
    'classes':          list(model.classes_),
    'vectorizer_vocab': len(vectorizer.vocabulary_),
    'ngram_range':      vectorizer.ngram_range,
    'max_features':     vectorizer.max_features,
    'uses_title':       'review_comment_title' in df.columns,
}
joblib.dump(model_config, MODEL_DIR / 'etapa03_model_config.pkl')

# Salvar métricas com timestamp
metrics = {
    'trained_at':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'train_size':    len(X_train),
    'test_size':     len(X_test),
    'accuracy':      round(accuracy,    4),
    'f1_macro':      round(f1_macro,    4),
    'f1_weighted':   round(f1_weighted, 4),
}
pd.DataFrame([metrics]).to_csv(MODEL_DIR / 'etapa03_metrics.csv', index=False)

print(f"  ✓ etapa03_sentiment_model.pkl")
print(f"  ✓ etapa03_vectorizer.pkl")
print(f"  ✓ etapa03_model_config.pkl")
print(f"  ✓ etapa03_metrics.csv")
print(f"  Diretório: {MODEL_DIR}")

print("\n" + "=" * 80)
print("  ETAPA 03 CONCLUÍDA COM SUCESSO!")
print("=" * 80)