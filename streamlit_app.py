import streamlit as st
import pandas as pd
import unicodedata
import re
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Classificador de Artigos Científicos",
    layout="wide"
)

# =========================================================
# NAVEGAÇÃO
# =========================================================

pagina = st.sidebar.radio(
    "Navegação",
    ["Início", "Análise", "Métricas"]
)

# =========================================================
# FUNÇÕES
# =========================================================

def normalizar(texto):

    texto = str(texto)

    texto = unicodedata.normalize("NFKD", texto)

    texto = texto.encode("ASCII", "ignore").decode("ASCII")

    return texto.lower()


def preparar_termos(entrada):

    return set([
        t.strip().lower()
        for t in re.split(",|\n", entrada)
        if t.strip() != ""
    ])


def score_texto(texto, termos, peso):

    return sum(
        peso
        for termo in termos
        if termo in texto
    )


def score_negativo(texto, termos, peso):

    return sum(
        peso
        for termo in termos
        if termo in texto
    )

# =========================================================
# PÁGINA INICIAL
# =========================================================

if pagina == "Início":

    st.title("🌊 Classificador de Artigos Científicos")

    st.markdown("""

    ## 📌 Como funciona o sistema

    Este aplicativo utiliza mineração de texto para classificar artigos científicos em:

    - ✅ Incluir
    - ⚠️ Avaliar
    - ❌ Excluir

    O sistema utiliza:
    - palavras-chave positivas
    - palavras-chave negativas
    - pesos diferentes para:
        - abstract
        - título
        - keywords

    Isso ajuda a reduzir falsos positivos durante revisões sistemáticas.

    ---

    ## 🚀 Como usar

    ### 1. Vá para a aba "Análise"

    ### 2. Faça upload da planilha CSV

    Estrutura esperada:

    | Coluna | Conteúdo |
    |---|---|
    | A | Abstract |
    | B | Título |
    | C | Keywords |

    ---

    ### 3. Insira termos relevantes

    Exemplo:
    - fish ecology
    - marine ecosystem
    - manguezal

    ---

    ### 4. Insira termos negativos

    Exemplo:
    - freshwater
    - river
    - fluvial

    Esses termos diminuem o score do artigo.

    ---

    ### 5. Rode a classificação

    O sistema irá:
    - calcular score
    - classificar artigos
    - mostrar métricas
    - permitir download do resultado

    """)

# =========================================================
# PÁGINA DE ANÁLISE
# =========================================================

elif pagina == "Análise":

    st.title("📊 Análise de Artigos")

    # =====================================================
    # UPLOAD
    # =====================================================

    file = st.file_uploader(
        "📁 Upload da planilha CSV",
        type=["csv"]
    )

    # =====================================================
    # TERMOS POSITIVOS
    # =====================================================

    st.markdown("## 🔑 Termos relevantes")

    termos_input = st.text_area(
        "Edite ou cole os termos relevantes",
        value="""fish community, fish assemblage, fish ecology
comunidade de peixes, ecologia de peixes, ictiofauna
marine ecosystem, ambiente marinho, oceano, estuario, manguezal
neotropical, tropical atlantic, gulf of mexico, caribbean sea"""
    )

    termos_relevantes = preparar_termos(
        termos_input
    )

    # =====================================================
    # TERMOS NEGATIVOS
    # =====================================================

    st.markdown("## 🚫 Termos para reduzir score")

    termos_negativos_input = st.text_area(
        "Artigos contendo esses termos terão score reduzido",
        value="""freshwater, river, rio, fluvial, agua doce"""
    )

    termos_negativos = preparar_termos(
        termos_negativos_input
    )

    # =====================================================
    # CLASSIFICAÇÃO
    # =====================================================

    def classificar(row):

        abstract = row[col_abstract]
        titulo = row[col_titulo]
        keywords = row[col_keywords]

        # =============================
        # SCORE POSITIVO
        # =============================

        score_positivo = (

            score_texto(
                abstract,
                termos_relevantes,
                1
            )

            +

            score_texto(
                titulo,
                termos_relevantes,
                2
            )

            +

            score_texto(
                keywords,
                termos_relevantes,
                3
            )
        )

        # =============================
        # SCORE NEGATIVO
        # =============================

        score_negativo_total = (

            score_negativo(
                abstract,
                termos_negativos,
                1
            )

            +

            score_negativo(
                titulo,
                termos_negativos,
                2
            )

            +

            score_negativo(
                keywords,
                termos_negativos,
                3
            )

        )

        # limitar penalização
        score_negativo_total = min(
            score_negativo_total,
            6
        )

        # =============================
        # SCORE FINAL
        # =============================

        score = (
            score_positivo -
            score_negativo_total
        )

        # =============================
        # CLASSIFICAÇÃO
        # =============================

        if score >= 6:

            classe = "Incluir"

        elif score >= 2:

            classe = "Avaliar"

        else:

            classe = "Excluir"

        return pd.Series([
            classe,
            score,
            score_positivo,
            score_negativo_total
        ])

    # =====================================================
    # EXECUÇÃO
    # =====================================================

    if file:

        df = pd.read_csv(
            file,
            encoding="latin1"
        )

        # =================================================
        # VALIDAÇÃO
        # =================================================

        if len(df.columns) < 3:

            st.error(
                "❌ A planilha precisa ter pelo menos 3 colunas."
            )

            st.stop()

        col_abstract = df.columns[0]
        col_titulo = df.columns[1]
        col_keywords = df.columns[2]

        # =================================================
        # PREVIEW
        # =================================================

        st.markdown("## 👀 Preview")

        st.dataframe(df.head())

        # =================================================
        # RODAR
        # =================================================

        if st.button("🚀 Rodar classificação"):

            # =============================================
            # NORMALIZAR TEXTO
            # =============================================

            df[col_abstract] = df[col_abstract].apply(
                normalizar
            )

            df[col_titulo] = df[col_titulo].apply(
                normalizar
            )

            df[col_keywords] = df[col_keywords].apply(
                normalizar
            )

            # =============================================
            # CLASSIFICAR
            # =============================================

            df[
                [
                    "cluster",
                    "score_final",
                    "score_positivo",
                    "score_negativo"
                ]
            ] = df.apply(
                classificar,
                axis=1
            )

            st.success(
                "✅ Classificação concluída!"
            )

            # =============================================
            # RESULTADO
            # =============================================

            st.markdown("## 📄 Resultado")

            st.dataframe(df)

            # =============================================
            # DISTRIBUIÇÃO
            # =============================================

            st.markdown(
                "## 📊 Distribuição por categoria"
            )

            counts = (
                df["cluster"]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Categoria",
                "Quantidade"
            ]

            st.dataframe(counts)

            # =============================================
            # DOWNLOAD
            # =============================================

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Baixar resultado",
                csv,
                "artigos_classificados.csv",
                "text/csv"
            )

# =========================================================
# PÁGINA MÉTRICAS
# =========================================================

elif pagina == "Métricas":

    st.title("📈 Avaliação do Classificador")

    st.markdown("""

    Faça upload de uma planilha contendo:

    - classificação REAL feita manualmente
    - classificação do robô

    ---

    ## 📋 Estrutura esperada

    A planilha deve conter:

    - cluster_real
    - cluster

    Exemplo:

    | cluster_real | cluster |
    |---|---|
    | Incluir | Incluir |
    | Excluir | Avaliar |

    """)

    file_metricas = st.file_uploader(
        "📁 Upload da planilha de validação",
        type=["csv"],
        key="metricas"
    )

    if file_metricas:

        df_metricas = pd.read_csv(
            file_metricas
        )

        # =================================================
        # VALIDAÇÃO
        # =================================================

        if "cluster_real" not in df_metricas.columns:

            st.error(
                "❌ Coluna 'cluster_real' não encontrada."
            )

            st.stop()

        if "cluster" not in df_metricas.columns:

            st.error(
                "❌ Coluna 'cluster' não encontrada."
            )

            st.stop()

        y_true = df_metricas["cluster_real"]
        y_pred = df_metricas["cluster"]

        # =================================================
        # MÉTRICAS
        # =================================================

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        # =================================================
        # RESULTADOS
        # =================================================

        st.markdown(
            "## 📊 Métricas principais"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Accuracy",
                f"{accuracy:.2%}"
            )

            st.metric(
                "Precision",
                f"{precision:.2%}"
            )

        with col2:

            st.metric(
                "Recall",
                f"{recall:.2%}"
            )

            st.metric(
                "F1-Score",
                f"{f1:.2%}"
            )

        # =================================================
        # INTERPRETAÇÃO
        # =================================================

        st.markdown("## 📚 Como interpretar")

        st.markdown("""

### 🎯 Accuracy (Acurácia)

Percentual total de acertos.

- ✅ > 85% → excelente
- ✅ 75–85% → bom
- ⚠️ < 70% → revisar termos

---

### 🔎 Precision

Mostra quantos artigos classificados como relevantes realmente eram relevantes.

- ✅ > 80% → poucos falsos positivos
- ⚠️ < 70% → muitos artigos irrelevantes incluídos

Se estiver baixa:
- adicione termos negativos
- aumente penalização
- remova termos genéricos

---

### 🧠 Recall

Capacidade do robô encontrar artigos relevantes.

Métrica MAIS importante em revisão sistemática.

- ✅ > 90% → excelente
- ✅ 80–90% → bom
- ⚠️ < 80% → risco de perder artigos importantes

Se estiver baixo:
- adicione sinônimos
- reduza penalização
- diminua threshold

---

### ⚖️ F1-Score

Equilíbrio entre precision e recall.

- ✅ > 85% → excelente
- ✅ 75–85% → bom
- ⚠️ < 70% → sistema instável

---

## 📌 Valores ideais

| Métrica | Ideal |
|---|---|
| Accuracy | > 80% |
| Precision | > 75% |
| Recall | > 90% |
| F1-score | > 80% |

""")

        # =================================================
        # MATRIZ DE CONFUSÃO
        # =================================================

        st.markdown(
            "## 🔍 Matriz de confusão"
        )

        labels = sorted(
            list(
                set(y_true) |
                set(y_pred)
            )
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=labels
        )

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        im = ax.imshow(cm)

        ax.set_xticks(
            np.arange(len(labels))
        )

        ax.set_yticks(
            np.arange(len(labels))
        )

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        plt.xlabel("Predito")
        plt.ylabel("Real")

        for i in range(len(labels)):
            for j in range(len(labels)):

                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

        st.pyplot(fig)

        # =================================================
        # ERROS
        # =================================================

        st.markdown(
            "## ❌ Artigos classificados incorretamente"
        )

        erros = df_metricas[
            df_metricas["cluster_real"] !=
            df_metricas["cluster"]
        ]

        st.dataframe(erros)

        # =================================================
        # DOWNLOAD
        # =================================================

        csv_erros = erros.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Baixar artigos com erro",
            csv_erros,
            "artigos_com_erro.csv",
            "text/csv"
        )
